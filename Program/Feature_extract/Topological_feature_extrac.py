import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Lipinski
from rdkit.Chem import rdPartialCharges
import networkx as nx
import pickle
import logging
from typing import List, Dict, Callable
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import argparse
import yaml

class TopologicalFeatureExtractor:
    def __init__(self, config: Dict):
        """
        Topological feature extractor with precise label assignment.

        Args:
            config (Dict): Configuration dictionary.
        """
        self.output_dir = os.path.abspath(config['output_dir'])
        os.makedirs(self.output_dir, exist_ok=True)

        # Configure logging system
        self.logger = logging.getLogger("TopologicalFeatureExtractor")
        self.logger.setLevel(config.get('log_level', 'INFO'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File logging
        log_file = os.path.join(self.output_dir, f"topo_feature_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console logging
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Define picklable descriptor calculation functions
        def _desc_molwt(mol): return Descriptors.MolWt(mol)

        def _desc_amide(mol): return rdMolDescriptors.CalcNumAmideBonds(mol)

        def _desc_kappa2(mol): return rdMolDescriptors.CalcKappa2(mol)

        def _desc_hdonors(mol): return Lipinski.NumHDonors(mol)

        def _desc_rotatable(mol): return Lipinski.NumRotatableBonds(mol)

        def _desc_aromatic(mol): return Lipinski.NumAromaticRings(mol)

        # Feature configuration
        self.feature_config = {
            "morgan_fp": {"radius": 3, "nBits": 512, "useFeatures": False},
            "key_descriptors": {
                "MolWt": _desc_molwt,
                "NumAmideBonds": _desc_amide,
                "Kappa2": _desc_kappa2,
                "NHOHCount": _desc_hdonors,
                "NumRotatableBonds": _desc_rotatable,
                "NumAromaticRings": _desc_aromatic
            },
            "functional_groups": {
                "hydrophobic": "[C;H3,H4]",
                "positive_charge": "[N;+1,+2]",
                "negative_charge": "[O;-]"
            },
            "feature_order": None  # Will be determined on first run
        }

        # Precise identifier definitions
        self.positive_prefixes = config.get('positive_prefixes', [">ACVP_", ">IACVP_"])
        self.negative_prefixes = config.get('negative_prefixes', [">nACVP_", ">InACVP_"])

        # Feature selector and scaler
        self.selector = VarianceThreshold(threshold=0.01)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _get_molecular_graph(self, mol) -> nx.Graph:
        """Construct a molecular graph containing only covalent bonds."""
        G = nx.Graph()
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType().name
            if bond_type in ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']:
                G.add_edge(
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond_type=bond_type
                )
        return G

    def _extract_sequence_features(self, sequence: str) -> Dict:
        """Extract features from an amino acid sequence."""
        return {
            "Cys_Pairs": sequence.count("C") // 2,
            "Proline_Kinks": sequence.count("P"),
            "Charge_Density": sum(1 for aa in sequence if aa in ["R", "K", "D", "E"]) / max(1, len(sequence)),
            "Hydrophobic_Ratio": sum(1 for aa in sequence if aa in ["A", "V", "L", "I", "M", "F", "W", "Y"]) / max(1, len(sequence))
        }

    def _extract_molecular_features(self, mol) -> Dict:
        """Extract features from an RDKit molecule object."""
        features = {}

        # 1. Morgan fingerprint
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.feature_config["morgan_fp"]["radius"],
                nBits=self.feature_config["morgan_fp"]["nBits"],
                useFeatures=self.feature_config["morgan_fp"]["useFeatures"]
            )
            features.update({f"morgan_{i}": int(b) for i, b in enumerate(fp)})
        except Exception as e:
            features.update({f"morgan_{i}": 0 for i in range(self.feature_config["morgan_fp"]["nBits"])})
            self.logger.warning(f"Morgan fingerprint calculation failed: {str(e)}")

        # 2. Molecular descriptors
        for desc_name, desc_func in self.feature_config["key_descriptors"].items():
            try:
                features[desc_name] = desc_func(mol)
            except Exception as e:
                features[desc_name] = 0
                self.logger.warning(f"Descriptor {desc_name} calculation failed: {str(e)}")

        # 3. Functional group counts
        for group_name, smarts in self.feature_config["functional_groups"].items():
            try:
                pat = Chem.MolFromSmarts(smarts)
                features[f"fg_{group_name}"] = len(mol.GetSubstructMatches(pat)) if pat else 0
            except Exception as e:
                features[f"fg_{group_name}"] = 0
                self.logger.warning(f"Functional group {group_name} matching failed: {str(e)}")

        # 4. Molecular graph features
        try:
            G = self._get_molecular_graph(mol)
            features.update({
                "topo_avg_degree": np.mean([d for _, d in G.degree()]) if G.nodes() else 0,
                "topo_cycle_count": len(list(nx.cycle_basis(G))),
                "topo_bridge_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            })
        except Exception as e:
            features.update({
                "topo_avg_degree": 0,
                "topo_cycle_count": 0,
                "topo_bridge_atoms": 0
            })
            self.logger.warning(f"Molecular graph feature calculation failed: {str(e)}")

        return features

    def _get_feature_order(self):
        """Generate a list of feature order."""
        if self.feature_config["feature_order"] is None:
            self.feature_config["feature_order"] = (
                    [f"morgan_{i}" for i in range(self.feature_config["morgan_fp"]["nBits"])] +
                    list(self.feature_config["key_descriptors"].keys()) +
                    [f"fg_{g}" for g in self.feature_config["functional_groups"]] +
                    ["topo_avg_degree", "topo_cycle_count", "topo_bridge_atoms"] +
                    ["Cys_Pairs", "Proline_Kinks", "Charge_Density", "Hydrophobic_Ratio"]
            )
        return self.feature_config["feature_order"]

    def _process_single_file(self, fasta_path: str, output_filename: str):
        """Process a single FASTA file and return features and labels."""
        sequences = []
        labels = []

        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # Precise label assignment logic
                    if any(line.startswith(prefix) for prefix in self.positive_prefixes):
                        current_label = 1
                        self.logger.debug(f"Identified as positive sample: {line}")
                    elif any(line.startswith(prefix) for prefix in self.negative_prefixes):
                        current_label = 0
                        self.logger.debug(f"Identified as negative sample: {line}")
                    else:
                        # If it doesn't match either positive or negative format, issue a warning and use default label
                        default_label = 1 if "pos" in output_filename.lower() else 0
                        self.logger.warning(f"Unrecognized header format: {line} | Using default label: {default_label}")
                        current_label = default_label
                elif line:  # Non-empty line and not starting with > is a sequence
                    sequences.append(line)
                    labels.append(current_label)

        # Validate label assignment
        unique_labels = set(labels)
        if len(unique_labels) != 2 and len(sequences) > 0:
            self.logger.error(f"Warning: Only one label found in file {fasta_path}: {unique_labels}")

        features = []
        feature_order = self._get_feature_order()
        failed_count = 0

        for seq in sequences:
            try:
                mol = Chem.MolFromSequence(seq)
                if not mol:
                    raise ValueError("Invalid peptide sequence")

                Chem.AddHs(mol)
                mol_feats = self._extract_molecular_features(mol)
                seq_feats = self._extract_sequence_features(seq)

                # Combine features and maintain fixed order
                combined = {**mol_feats, **seq_feats}
                features.append([combined.get(k, 0) for k in feature_order])

            except Exception as e:
                failed_count += 1
                self.logger.error(f"Failed to process sequence: {seq[:10]}... | Error: {str(e)}")
                features.append(np.zeros(len(feature_order)))

        feature_array = np.array(features, dtype=np.float32)
        label_array = np.array(labels, dtype=np.int32)

        # Save results
        feature_output_path = os.path.join(self.output_dir, output_filename)
        label_output_path = os.path.join(self.output_dir, output_filename.replace("features", "labels"))

        np.save(feature_output_path, feature_array)
        np.save(label_output_path, label_array)

        self.logger.info(
            f"Processing complete | File: {os.path.basename(fasta_path)} | Success: {len(sequences) - failed_count} | "
            f"Failed: {failed_count} | Feature dimension: {feature_array.shape[1]} | "
            f"Label distribution - Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}"
        )

    def fit_transform(self, train_datasets: Dict[str, str]):
        """
        Fit feature selector and scaler on training data.

        Args:
            train_datasets: Dictionary of training dataset paths {name: path}
        """
        if self.is_fitted:
            return

        self.logger.info("Starting to fit feature selector and scaler on training data...")

        # Collect all training features
        train_features = []
        for name, path in train_datasets.items():
            output_filename = f"{name}_features.npy"
            self._process_single_file(path, output_filename)

            # Load the just-saved features
            feature_path = os.path.join(self.output_dir, output_filename)
            train_features.append(np.load(feature_path))

        # Combine all training features
        X_train = np.concatenate(train_features)

        # Fit feature selector and scaler
        X_train_selected = self.selector.fit_transform(X_train)
        self.scaler.fit(X_train_selected)

        self.is_fitted = True
        self.logger.info(
            f"Feature selector and scaler fitting complete | Original features: {X_train.shape[1]} | "
            f"Selected features: {X_train_selected.shape[1]}"
        )

    def transform(self, dataset_name: str, dataset_path: str):
        """
        Apply feature transformation to a dataset.

        Args:
            dataset_name: Dataset name
            dataset_path: Dataset path
        """
        if not self.is_fitted:
            raise RuntimeError("Feature extractor not fitted, please call fit_transform() first")

        # Process dataset
        output_filename = f"{dataset_name}_features.npy"
        self._process_single_file(dataset_path, output_filename)

        # Load features and apply transformation
        feature_path = os.path.join(self.output_dir, output_filename)
        features = np.load(feature_path)

        # Apply feature selection and scaling
        features_selected = self.selector.transform(features)
        features_scaled = self.scaler.transform(features_selected)

        # Save transformed features
        output_path = os.path.join(self.output_dir, f"{dataset_name}_features_transformed.npy")
        np.save(output_path, features_scaled)
        self.logger.info(f"Saved transformed features to: {output_path}")

    def save_extractor(self):
        """Save feature extractor configuration (picklable version)"""
        save_path = os.path.join(self.output_dir, "topo_feature_extractor.pkl")

        # Create a picklable configuration dictionary
        save_config = {
            "feature_config": {
                "morgan_fp": self.feature_config["morgan_fp"],
                "key_descriptors": list(self.feature_config["key_descriptors"].keys()),
                "functional_groups": self.feature_config["functional_groups"],
                "feature_order": self._get_feature_order()
            },
            "positive_prefixes": self.positive_prefixes,
            "negative_prefixes": self.negative_prefixes,
            "selector": self.selector,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted
        }

        with open(save_path, "wb") as f:
            pickle.dump(save_config, f)
        self.logger.info(f"Feature extractor configuration saved to: {save_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Feature Extractor")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Initialize extractor
    extractor = TopologicalFeatureExtractor(config)

    # Define dataset paths
    train_datasets = config['train_datasets']
    test_datasets = config['test_datasets']
    independent_datasets = config['independent_datasets']

    # 1. Fit feature selector and scaler on training data
    extractor.fit_transform(train_datasets)

    # 2. Transform training data
    for name, path in train_datasets.items():
        extractor.transform(name, path)

    # 3. Transform test data
    for name, path in test_datasets.items():
        extractor.transform(name, path)

    # 4. Transform independent datasets
    for name, path in independent_datasets.items():
        extractor.transform(name, path)

    # Save extractor state
    extractor.save_extractor()