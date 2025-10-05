import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (SelectKBest, mutual_info_classif,
                                       VarianceThreshold, SelectFromModel)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
import pandas as pd
import pickle
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModalityEmbedding(nn.Module):
    """Modality embedding layer"""

    def __init__(self, input_dims, embed_size=128):
        super(ModalityEmbedding, self).__init__()
        self.embed_layers = nn.ModuleDict()
        for mod, dim in input_dims.items():
            self.embed_layers[mod] = nn.Sequential(
                nn.Linear(dim, embed_size),
                nn.ReLU(),
                nn.LayerNorm(embed_size)
            )

    def forward(self, modality_features):
        embedded = {}
        for mod, feat in modality_features.items():
            if len(feat.shape) == 1:
                feat = feat.unsqueeze(0)
            embedded[mod] = self.embed_layers[mod](feat)
        return embedded


class ModalityAttention(nn.Module):
    """Modality attention mechanism"""

    def __init__(self, embed_size):
        super(ModalityAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Tanh(),
            nn.Linear(embed_size, 1, bias=False))

    def forward(self, embedded_features):
        stacked = torch.stack(list(embedded_features.values()), dim=1)
        attn_scores = self.attention(stacked)
        attn_weights = F.softmax(attn_scores, dim=1)
        fused = (stacked * attn_weights).sum(dim=1)
        return fused, attn_weights.squeeze(-1)


class CrossModalityInteraction(nn.Module):
    """Cross-modality interaction layer"""

    def __init__(self, embed_size):
        super(CrossModalityInteraction, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            min_len = min(x1.shape[1], x2.shape[1])
            x1 = x1[:, :min_len]
            x2 = x2[:, :min_len]

        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return x1 + self.gamma * out


class MultiModalFusionModel(nn.Module):
    """Multi-modal fusion model"""

    def __init__(self, input_dims, embed_size=128):
        super(MultiModalFusionModel, self).__init__()
        self.embed_size = embed_size
        self.embedding = ModalityEmbedding(input_dims, embed_size)
        self.attention = ModalityAttention(embed_size)

        # Define modality groups
        self.structure_mods = ['secondary', 'tertiary', 'graph']
        self.sequence_mods = ['sequence', 'evolution']
        self.topology_mods = ['topology']

        self.cross_interaction = CrossModalityInteraction(embed_size)
        self.proj_input_dim = embed_size * 4
        self.proj = self._build_projection_layers()

    def _build_projection_layers(self):
        return nn.Sequential(
            nn.Linear(self.proj_input_dim, self.embed_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.embed_size * 2),
            nn.Linear(self.embed_size * 2, self.embed_size),
            nn.ReLU(),
            nn.LayerNorm(self.embed_size)
        )

    def forward(self, modality_features):
        embedded = self.embedding(modality_features)
        fused, attn_weights = self.attention(embedded)

        group_features = []
        for group in [self.structure_mods, self.sequence_mods, self.topology_mods]:
            group_feats = [embedded.get(mod, None) for mod in group]
            group_feats = [f for f in group_feats if f is not None]
            if group_feats:
                group_mean = torch.stack(group_feats, dim=0).mean(dim=0)
                group_features.append(group_mean)

        if len(group_features) >= 2:
            interacted1 = self.cross_interaction(group_features[0].unsqueeze(1),
                                                 group_features[1].unsqueeze(1))
            interacted2 = self.cross_interaction(group_features[1].unsqueeze(1),
                                                 group_features[0].unsqueeze(1))
            group_features = [interacted1.squeeze(1), interacted2.squeeze(1)] + group_features[2:]

        all_features = [fused]
        all_features.extend(group_features)

        min_len = min([f.shape[-1] for f in all_features])
        all_features = [f[..., :min_len] for f in all_features]

        concatenated = torch.cat(all_features, dim=-1)
        if concatenated.shape[-1] != self.proj_input_dim:
            self.proj = nn.Sequential(
                nn.Linear(concatenated.shape[-1], self.proj_input_dim),
                nn.ReLU(),
                *list(self.proj.children())[1:]
            ).to(concatenated.device)

        output = self.proj(concatenated)
        return output


class MultiModalFeatureProcessor:
    """Multi-modal feature processor"""

    def __init__(self, modality_paths, output_dir="./feature_selection_results"):
        self.modality_paths = modality_paths
        self.output_dir = output_dir
        self.modality_data = None
        self.selected_features = None

    def load_modality(self, modality_path, dataset_type='train'):
        """Load modality data"""
        try:
            pos_feat = np.load(os.path.join(modality_path, f"{dataset_type}_pos_features.npy"))
            neg_feat = np.load(os.path.join(modality_path, f"{dataset_type}_neg_features.npy"))
            pos_label = np.load(os.path.join(modality_path, f"{dataset_type}_pos_labels.npy"))
            neg_label = np.load(os.path.join(modality_path, f"{dataset_type}_neg_labels.npy"))

            pos_feat = pos_feat.reshape(-1, pos_feat.shape[-1]) if len(pos_feat.shape) > 2 else pos_feat
            neg_feat = neg_feat.reshape(-1, neg_feat.shape[-1]) if len(neg_feat.shape) > 2 else neg_feat

            print(f"\n{modality_path} - {dataset_type}:")
            print(f"Positive samples: {pos_feat.shape}, Negative samples: {neg_feat.shape}")

            min_pos_len = min(len(pos_feat), len(pos_label))
            min_neg_len = min(len(neg_feat), len(neg_label))

            pos_feat = pos_feat[:min_pos_len]
            pos_label = pos_label[:min_pos_len]
            neg_feat = neg_feat[:min_neg_len]
            neg_label = neg_label[:min_neg_len]

            X = np.vstack([pos_feat, neg_feat])
            y = np.hstack([pos_label, neg_label])

            print(f"Merged: X.shape={X.shape}, y.shape={y.shape}")
            return X, y
        except Exception as e:
            print(f"Failed to load {modality_path} {dataset_type} data: {str(e)}")
            return None, None

    def load_all_modalities(self):
        """Load all modality data"""
        modality_data = {}
        dataset_types = ['train', 'test', 'independent']

        min_samples = {dt: float('inf') for dt in dataset_types}

        for mod_path in self.modality_paths.values():
            for dt in dataset_types:
                X, _ = self.load_modality(mod_path, dt)
                if X is not None:
                    min_samples[dt] = min(min_samples[dt], X.shape[0])

        for mod_name, mod_path in self.modality_paths.items():
            mod_data = {}
            scaler = None

            for dt in dataset_types:
                X, y = self.load_modality(mod_path, dt)
                if X is not None:
                    X = X[:min_samples[dt]]
                    y = y[:min_samples[dt]]

                    if dt == 'train':
                        scaler = StandardScaler()
                        X = scaler.fit_transform(X)
                    else:
                        X = scaler.transform(X)

                    mod_data[dt] = (X, y)
                else:
                    mod_data[dt] = (None, None)

            modality_data[mod_name] = mod_data
            print(f"Loaded {mod_name} modality | Train: {mod_data['train'][0].shape} | "
                  f"Test: {mod_data['test'][0].shape} | Independent: {mod_data['independent'][0].shape}")

        self.modality_data = modality_data
        return modality_data

    def select_features(self, mod_name, X_train, y_train, X_test, X_independent=None):
        """Feature selection for each modality"""
        try:
            if mod_name == 'sequence':
                selector = Pipeline([
                    ('variance', VarianceThreshold(threshold=0.1)),
                    ('mutual_info', SelectKBest(mutual_info_classif, k=min(50, X_train.shape[1])))
                ])
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                X_independent_selected = selector.transform(X_independent) if X_independent is not None else None

            elif mod_name == 'topology':
                selector = PCA(n_components=0.95, random_state=42)
                X_train_selected = selector.fit_transform(X_train)
                X_test_selected = selector.transform(X_test)
                X_independent_selected = selector.transform(X_independent) if X_independent is not None else None

            elif mod_name == 'evolution':
                selector = SelectFromModel(
                    RandomForestClassifier(n_estimators=100, random_state=42),
                    threshold='median')
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                X_independent_selected = selector.transform(X_independent) if X_independent is not None else None

            else:
                selector = VarianceThreshold(threshold=0.1)
                X_train_selected = selector.fit_transform(X_train)
                X_test_selected = selector.transform(X_test)
                X_independent_selected = selector.transform(X_independent) if X_independent is not None else None

            print(f"{mod_name} modality selected dimensions: {X_train_selected.shape[1]}")
            return (X_train_selected, X_test_selected, X_independent_selected)
        except Exception as e:
            print(f"{mod_name} modality selection failed, using original features: {str(e)}")
            return (X_train, X_test, X_independent)

    def perform_feature_selection(self):
        """Perform feature selection for all modalities"""
        print("\n=== Single Modality Feature Selection ===")
        selected_features = {}

        for mod_name, mod_data in self.modality_data.items():
            X_train, y_train = mod_data['train']
            X_test, y_test = mod_data['test']
            X_independent, y_independent = mod_data['independent']

            selected = self.select_features(mod_name, X_train, y_train, X_test, X_independent)
            selected_features[mod_name] = {
                'train': (selected[0], y_train),
                'test': (selected[1], y_test),
                'independent': (selected[2], y_independent)
            }

        self.selected_features = selected_features
        return selected_features

    def fuse_features(self, fusion_order, embed_size=128, epochs=30, batch_size=32):
        """Fuse features using attention mechanism"""
        X_train_dict = {mod: self.selected_features[mod]['train'][0] for mod in fusion_order
                        if mod in self.selected_features}
        y_train = self.selected_features[fusion_order[0]]['train'][1]

        input_dims = {mod: X_train_dict[mod].shape[1] for mod in X_train_dict}
        train_tensors = {mod: torch.FloatTensor(X_train_dict[mod]) for mod in X_train_dict}

        model = MultiModalFusionModel(input_dims, embed_size).to(device)
        print(f"Fusion model parameters: embed_size={embed_size}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(*[train_tensors[mod] for mod in train_tensors],
                                      torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch_features = {mod: batch[i].to(device) for i, mod in enumerate(train_tensors)}
                labels = batch[-1].to(device).float()

                optimizer.zero_grad()
                fused = model(batch_features)
                loss = criterion(fused.mean(dim=1), labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

        fused_features = {}
        for dataset_type in ['train', 'test', 'independent']:
            X_dict = {mod: torch.FloatTensor(self.selected_features[mod][dataset_type][0])
                      for mod in fusion_order if mod in self.selected_features}

            with torch.no_grad():
                fused = []
                for i in range(0, len(X_dict[fusion_order[0]]), batch_size):
                    batch = {mod: X_dict[mod][i:i + batch_size].to(device) for mod in X_dict}
                    fused_batch = model(batch)
                    fused.append(fused_batch.cpu().numpy())

                X_fused = np.concatenate(fused, axis=0)
                y = self.selected_features[fusion_order[0]][dataset_type][1]
                fused_features[dataset_type] = (X_fused, y)

        print(f"\nFused dimensions: {fused_features['train'][0].shape[1]}")
        return fused_features

    def dynamic_feature_selection(self, fusion_order, min_features=10, max_features=30):
        """Dynamic feature selection"""
        X_train_all, X_test_all, X_independent_all = [], [], []
        support_mask = []
        feature_sources = []
        selected_counts = {}

        first_mod = fusion_order[0]
        y_train = self.selected_features[first_mod]['train'][1]
        y_test = self.selected_features[first_mod]['test'][1]
        y_independent = self.selected_features[first_mod]['independent'][1]

        for mod in fusion_order:
            if mod in self.selected_features:
                X_train, _ = self.selected_features[mod]['train']
                X_test, _ = self.selected_features[mod]['test']
                X_independent, _ = self.selected_features[mod]['independent']

                n_features = X_train.shape[1]
                n_select = min(
                    max(min_features, int(n_features * 0.3)),
                    max_features
                )
                selected_counts[mod] = n_select

                selector = SelectKBest(mutual_info_classif, k=min(n_select, n_features))
                X_train_sel = selector.fit_transform(X_train, y_train)
                X_test_sel = selector.transform(X_test)
                X_independent_sel = selector.transform(X_independent)

                X_train_all.append(X_train_sel)
                X_test_all.append(X_test_sel)
                X_independent_all.append(X_independent_sel)

                support_mask.extend([True] * X_train_sel.shape[1] + [False] * (n_features - X_train_sel.shape[1]))
                feature_sources.extend([mod] * n_features)

                print(f"{mod} modality: {n_features} features -> {X_train_sel.shape[1]} selected")

        print("\nFeature selection statistics:")
        for mod, count in selected_counts.items():
            print(f"{mod} modality: {count} features selected")

        total_selected = sum(selected_counts.values())
        print(f"\nTotal features selected: {total_selected}")

        return (
            np.hstack(X_train_all),
            np.hstack(X_test_all),
            np.hstack(X_independent_all),
            y_train,
            y_test,
            y_independent,
            np.array(support_mask),
            feature_sources,
            selected_counts
        )

    def save_results(self, X_train, X_test, X_independent,
                     y_train, y_test, y_independent,
                     selected_sources, full_sources,
                     selected_indices, selected_counts):
        """Save feature selection results"""

        def check_samples(X, y, name):
            if len(X) != len(y):
                raise ValueError(
                    f"{name} sample mismatch: X({len(X)}) != y({len(y)})\n"
                    "Please check data processing steps"
                )

        check_samples(X_train, y_train, 'Training set')
        check_samples(X_test, y_test, 'Test set')
        check_samples(X_independent, y_independent, 'Independent set')

        os.makedirs(self.output_dir, exist_ok=True)

        np.save(os.path.join(self.output_dir, 'X_train_selected.npy'), X_train)
        np.save(os.path.join(self.output_dir, 'X_test_selected.npy'), X_test)
        np.save(os.path.join(self.output_dir, 'X_independent_selected.npy'), X_independent)

        np.save(os.path.join(self.output_dir, 'y_train.npy'), y_train[:len(X_train)])
        np.save(os.path.join(self.output_dir, 'y_test.npy'), y_test[:len(X_test)])
        np.save(os.path.join(self.output_dir, 'y_independent.npy'), y_independent[:len(X_independent)])

        pd.DataFrame({
            'feature_index': np.arange(len(full_sources)),
            'source_modality': full_sources,
            'selected': selected_indices
        }).to_csv(os.path.join(self.output_dir, 'feature_source_mapping.csv'), index=False)

        metadata = {
            'selected_feature_sources': selected_sources,
            'all_feature_sources': full_sources,
            'selected_counts': selected_counts,
            'feature_dimensions': X_train.shape[1],
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'independent_samples': X_independent.shape[0]
        }

        with open(os.path.join(self.output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        print(f"\nFeature selection results saved to {self.output_dir}:")
        print("Actual feature counts per modality:")
        for mod, count in selected_counts.items():
            print(f"  {mod}: {count} features")


def main():
    """Main execution function"""

    # Update these paths according to your data structure
    modality_paths = {
        'sequence': './data/Seq_Feature',
        'secondary': './data/Struc_Feature/Secondary_Feature',
        'tertiary': './data/Struc_Feature/Tertiary_Feature',
        'graph': './data/Struc_Feature/Graph_Feature',
        'topology': './data/Topo_Feature',
        'evolution': './data/Evolu_Feature'
    }

    output_dir = "./feature_selection_results"
    fusion_order = ['sequence', 'secondary', 'tertiary', 'graph', 'topology', 'evolution']

    # Initialize processor
    processor = MultiModalFeatureProcessor(modality_paths, output_dir)

    # Load data
    print("\n=== Loading All Modality Data ===")
    processor.load_all_modalities()

    # Feature selection
    processor.perform_feature_selection()

    # Feature fusion
    print("\n=== Attention Feature Fusion ===")
    fused_features = processor.fuse_features(fusion_order)

    # Dynamic feature selection
    print("\n=== Dynamic Feature Selection ===")
    results = processor.dynamic_feature_selection(fusion_order)

    X_train_final, X_test_final, X_independent_final, y_train_final, y_test_final, y_independent_final, selected_indices, full_sources, selected_counts = results

    # Track feature sources
    selected_sources = [full_sources[i] for i in range(len(selected_indices)) if selected_indices[i]]

    print("\n=== Feature Sources ===")
    for mod in set(selected_sources):
        print(f"{mod} modality: {selected_sources.count(mod)} features")

    # Save results
    processor.save_results(
        X_train_final, X_test_final, X_independent_final,
        y_train_final, y_test_final, y_independent_final,
        selected_sources, full_sources, selected_indices, selected_counts
    )


if __name__ == "__main__":
    main()