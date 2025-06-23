"""
Secondary structure feature extractor (cleaned version for GitHub release)
"""

import os
import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder

# 自动选择可用设备（不再强制指定 GPU ID）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SecondaryStructureFeatureExtractor:
    def __init__(self, data_root: str, output_dir: str):
        self.datasets = {
            'train_pos': os.path.join(data_root, 'train_pos'),
            'train_neg': os.path.join(data_root, 'train_neg'),
            'test_pos': os.path.join(data_root, 'test_pos'),
            'test_neg': os.path.join(data_root, 'test_neg'),
            'independent_pos': os.path.join(data_root, 'independent_pos'),
            'independent_neg': os.path.join(data_root, 'independent_neg')
        }
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.confidence_threshold = None
        self.short_peptide_length = 15
        self.fitted_ = False

        self.kd_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }

        self.position_encoder = OneHotEncoder(
            categories=[['none', 'N-term', 'middle', 'C-term']],
            handle_unknown='ignore'
        )

    def _determine_threshold(self, train_dfs):
        all_confidences = []
        for df in train_dfs:
            conf = df['p[q3_H]'] + df['p[q3_E]'] + df['p[q3_C]']
            all_confidences.extend(conf.values)
        self.confidence_threshold = np.median(all_confidences)
        print(f"[Threshold] Set to {self.confidence_threshold:.3f}")

    def _adjust_threshold(self, seq_length):
        if not self.fitted_:
            raise RuntimeError("Call fit() first.")
        if seq_length < self.short_peptide_length:
            return max(0.5, self.confidence_threshold - 0.02 * (self.short_peptide_length - seq_length))
        return self.confidence_threshold

    def _load_csv(self, path):
        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            required = ['seq', 'q3', 'p[q3_H]', 'p[q3_E]', 'p[q3_C]']
            return df if all(col in df.columns for col in required) else None
        except:
            return None

    def fit(self):
        if self.fitted_:
            print("Already fitted.")
            return

        print("Fitting feature extractor...")
        train_dfs = []
        for name in ['train_pos', 'train_neg']:
            for f in os.listdir(self.datasets[name]):
                if f.endswith('.csv'):
                    df = self._load_csv(os.path.join(self.datasets[name], f))
                    if df is not None:
                        train_dfs.append(df)

        self._determine_threshold(train_dfs)
        self.position_encoder.fit([['N-term'], ['middle'], ['C-term'], ['none']])
        self.fitted_ = True
        self._save_extractor()

    def _calculate_global_stats(self, df):
        threshold = self._adjust_threshold(len(df))
        valid_df = df[(df['p[q3_H]'] + df['p[q3_E]'] + df['p[q3_C]']) >= threshold]
        if len(valid_df) == 0:
            return dict.fromkeys(['H_percent', 'E_percent', 'C_percent', 'dominant_strength'], 0.0)
        return {
            'H_percent': (valid_df['q3'] == 'H').mean(),
            'E_percent': (valid_df['q3'] == 'E').mean(),
            'C_percent': (valid_df['q3'] == 'C').mean(),
            'dominant_strength': max(valid_df[['p[q3_H]', 'p[q3_E]', 'p[q3_C]']].mean())
        }

    def _calculate_fragment_features(self, df):
        ss_seq = df['q3'].values
        max_h = max_e = current_h = current_e = 0
        h_starts = []
        e_starts = []
        for i, ss in enumerate(ss_seq):
            if ss == 'H':
                current_h += 1; current_e = 0
                if current_h == 1: h_starts.append(i)
                max_h = max(max_h, current_h)
            elif ss == 'E':
                current_e += 1; current_h = 0
                if current_e == 1: e_starts.append(i)
                max_e = max(max_e, current_e)
            else:
                current_h = current_e = 0

        def _pos_feature(starts):
            if not starts: return 'none', 0.0
            pos = starts[0] / len(df)
            if pos < 0.33: return 'N-term', pos
            elif pos > 0.66: return 'C-term', pos
            else: return 'middle', pos

        h_pos, h_ratio = _pos_feature(h_starts)
        e_pos, e_ratio = _pos_feature(e_starts)

        def _cv(ss_type):
            lengths = []
            current = 0
            for ss in ss_seq:
                if ss == ss_type:
                    current += 1
                elif current > 0:
                    lengths.append(current)
                    current = 0
            if current > 0:
                lengths.append(current)
            return float(np.std(lengths) / np.mean(lengths)) if lengths else 0.0

        return {
            'max_H_length': max_h,
            'max_E_length': max_e,
            'H_position': h_pos,
            'E_position': e_pos,
            'H_position_ratio': h_ratio,
            'E_position_ratio': e_ratio,
            'H_length_CV': _cv('H'),
            'E_length_CV': _cv('E')
        }

    def _calculate_transition_matrix(self, df):
        ss_types = ['H', 'E', 'C']
        trans_counts = np.zeros((3, 3))
        for i in range(len(df) - 1):
            a, b = df.iloc[i]['q3'], df.iloc[i + 1]['q3']
            if a in ss_types and b in ss_types:
                trans_counts[ss_types.index(a), ss_types.index(b)] += 1

        row_sums = trans_counts.sum(axis=1, keepdims=True)
        trans_matrix = np.divide(trans_counts, row_sums, out=np.zeros_like(trans_counts), where=row_sums != 0)
        try:
            eigvals, eigvecs = np.linalg.eig(trans_matrix.T)
            stat = eigvecs[:, np.isclose(eigvals, 1)].real[:, 0]
            stat /= stat.sum()
        except:
            stat = np.ones(3) / 3

        features = {}
        for i, from_ in enumerate(ss_types):
            for j, to_ in enumerate(ss_types):
                features[f'trans_{from_}_to_{to_}'] = trans_matrix[i, j]
            features[f'stat_{from_}'] = stat[i]
        return features

    def _calculate_physicochemical(self, df):
        helix_hydro = []
        current = []
        for _, row in df.iterrows():
            if row['q3'] == 'H':
                current.append(row['seq'])
            elif current:
                helix_hydro.append(np.mean([self.kd_scale.get(aa, 0) for aa in current]))
                current = []
        if current:
            helix_hydro.append(np.mean([self.kd_scale.get(aa, 0) for aa in current]))

        polar = {'D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y'}
        polar_pos = [i for i, row in df.iterrows() if row['q3'] == 'E' and row['seq'] in polar]
        clustering = 1 / np.mean(np.diff(polar_pos)) if len(polar_pos) > 1 else 0.0

        return {
            'avg_helix_hydro': np.mean(helix_hydro) if helix_hydro else 0.0,
            'polar_clustering_sheet': clustering
        }

    def _encode_features(self, features):
        h_pos = features.pop('H_position')
        e_pos = features.pop('E_position')
        h_encoded = self.position_encoder.transform([[h_pos]]).toarray()[0]
        e_encoded = self.position_encoder.transform([[e_pos]]).toarray()[0]

        numeric = {k: float(v) if isinstance(v, (int, float)) else 0.0 for k, v in features.items()}
        for i, val in enumerate(h_encoded):
            numeric[f'H_position_{i}'] = val
        for i, val in enumerate(e_encoded):
            numeric[f'E_position_{i}'] = val
        return numeric

    def extract_features(self, csv_path):
        if not self.fitted_:
            raise RuntimeError("Call fit() before extracting features.")
        df = self._load_csv(csv_path)
        if df is None:
            return None
        features = {}
        features.update(self._calculate_global_stats(df))
        features.update(self._calculate_fragment_features(df))
        features.update(self._calculate_transition_matrix(df))
        features.update(self._calculate_physicochemical(df))
        return self._encode_features(features)

    def process_dataset(self, name):
        input_dir = self.datasets[name]
        features, labels = [], []
        for f in tqdm(os.listdir(input_dir), desc=f"{name}"):
            if f.endswith('.csv'):
                feats = self.extract_features(os.path.join(input_dir, f))
                if feats is not None:
                    features.append(list(feats.values()))
                    labels.append(1 if 'pos' in name else 0)
        np.save(os.path.join(self.output_dir, f"{name}_features.npy"), np.array(features))
        np.save(os.path.join(self.output_dir, f"{name}_labels.npy"), np.array(labels))

    def _save_extractor(self):
        joblib.dump(self, os.path.join(self.output_dir, "feature_extractor.pkl"))

    def process_all_datasets(self):
        self.fit()
        for name in self.datasets:
            self.process_dataset(name)


if __name__ == "__main__":
    extractor = SecondaryStructureFeatureExtractor(data_root="data/", output_dir="output/secondary")
    extractor.process_all_datasets()
