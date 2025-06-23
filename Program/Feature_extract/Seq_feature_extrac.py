import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy, moment
from itertools import product
import joblib
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import argparse
import yaml

# ==================== Physical Chemical Properties Definition ====================
class ACoVPPhysChemProps:
    """Physical chemical properties specific to anti-coronavirus peptides."""
    def __init__(self):
        self.properties = {
            'hydrophobicity': {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                               'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                               'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                               'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2},
            'net_charge': {'A': 0, 'R': +1, 'N': 0, 'D': -1, 'C': 0,
                           'E': -1, 'Q': 0, 'G': 0, 'H': +0.5, 'I': 0,
                           'L': 0, 'K': +1, 'M': 0, 'F': 0, 'P': 0,
                           'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0},
            'isoelectric_point': {'A': 6.00, 'R': 10.76, 'N': 5.41, 'D': 2.77, 'C': 5.07,
                                  'E': 3.22, 'Q': 5.65, 'G': 5.97, 'H': 7.59, 'I': 6.02,
                                  'L': 5.98, 'K': 9.74, 'M': 5.74, 'F': 5.48, 'P': 6.30,
                                  'S': 5.68, 'T': 5.87, 'W': 5.89, 'Y': 5.66, 'V': 5.96},
            'hydrophobic_moment': {'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
                                   'E': -0.74, 'Q': -0.85, 'G': 0.48, 'H': -0.40, 'I': 1.38,
                                   'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
                                   'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08},
            'berman_accessibility': {'A': 0.56, 'R': 0.55, 'N': 0.54, 'D': 0.51, 'C': 0.49,
                                     'E': 0.50, 'Q': 0.53, 'G': 0.57, 'H': 0.50, 'I': 0.47,
                                     'L': 0.53, 'K': 0.52, 'M': 0.50, 'F': 0.48, 'P': 0.58,
                                     'S': 0.55, 'T': 0.52, 'W': 0.48, 'Y': 0.50, 'V': 0.54},
            'molecular_weight': {'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
                                 'E': 147.13, 'Q': 146.15, 'G': 75.07, 'H': 155.16, 'I': 131.17,
                                 'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
                                 'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15},
            'aromaticity': {'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0,
                            'E': 0, 'Q': 0, 'G': 0, 'H': 1, 'I': 0,
                            'L': 0, 'K': 0, 'M': 0, 'F': 1, 'P': 0,
                            'S': 0, 'T': 0, 'W': 1, 'Y': 1, 'V': 0}
        }
        self.property_names = list(self.properties.keys())

    def get_property_values(self, sequence, prop_name):
        return [self.properties[prop_name].get(aa, 0) for aa in sequence]

# ==================== Core Feature Calculation ====================
def calculate_aac(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    counts = {aa: 0 for aa in amino_acids}
    for aa in sequence:
        if aa in counts:
            counts[aa] += 1
    total = max(1, sum(counts.values()))
    return [counts[aa] / total for aa in amino_acids]

def calculate_dpc(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    counts = {a + b: 0 for a, b in product(amino_acids, repeat=2)}
    if len(sequence) < 2:
        return [0.0] * 400
    total = len(sequence) - 1
    for i in range(total):
        dimer = sequence[i:i + 2]
        if dimer in counts:
            counts[dimer] += 1
    return [counts[d] / total for d in sorted(counts)]

def calculate_ctd(sequence, pc_props):
    features = []
    for prop_name in pc_props.property_names:
        values = pc_props.get_property_values(sequence, prop_name)
        diffs = np.abs(np.diff(values)) if len(values) > 1 else [0.0]
        features.extend([
            np.mean(values), np.std(values), np.min(values), np.max(values),
            values[-1] if len(values) > 0 else 0.0, np.mean(diffs),
            np.sum(diffs > 0.5), diffs[-1], *np.percentile(values, [5, 25, 50, 75, 95]),
            moment(values, moment=3)
        ])
    return features

def calculate_sequence_complexity(sequence, window=5):
    if len(sequence) < window:
        return [0.0] * 7
    entropies = []
    for i in range(len(sequence) - window + 1):
        sub_seq = sequence[i:i + window]
        counts = [sub_seq.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY']
        entropies.append(entropy(counts, base=2))
    return [
        np.mean(entropies), np.std(entropies), np.min(entropies),
        np.max(entropies), entropies[0], entropies[-1],
        len(set(sequence)) / len(sequence)
    ]

class ACoVPFeatureExtractor:
    def __init__(self, config: Dict):
        """
        Feature extractor for biological sequences.

        Args:
            config (Dict): Configuration dictionary.
        """
        self.output_dir = os.path.abspath(config['output_dir'])
        self.max_k = config['max_k']
        self.n_jobs = config['n_jobs']
        os.makedirs(self.output_dir, exist_ok=True)

        self.scaler = StandardScaler()
        self.dpc_mask = None
        self.kmer_vectorizer = None
        self.pc_props = ACoVPPhysChemProps()
        self.expected_features = None

    def _determine_k(self, length):
        if length <= 15: return 2
        elif length <= 30: return 3
        else: return min(4, self.max_k)

    def _parallel_extract(self, func, sequences):
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            return list(tqdm(executor.map(func, sequences), total=len(sequences)))

    def fit(self, train_sequences):
        # 1. Calculate and filter DPC features
        train_dpc = np.array(self._parallel_extract(calculate_dpc, train_sequences))
        dpc_freq = train_dpc.sum(axis=0)
        qualified = np.where(dpc_freq >= 1)[0]
        n_to_keep = min(200, len(qualified))
        self.dpc_mask = np.zeros(400, dtype=bool)
        if n_to_keep > 0:
            top_indices = qualified[np.argsort(-dpc_freq[qualified])[:n_to_keep]]
            self.dpc_mask[top_indices] = True
        else:
            self.dpc_mask[np.argsort(-dpc_freq)[:20]] = True

        # 2. Initialize k-mer vectorizer
        lengths = [len(seq) for seq in train_sequences]
        k = self._determine_k(np.median(lengths))
        self.kmer_vectorizer = TfidfVectorizer(
            ngram_range=(k, k), analyzer='char', max_features=50, lowercase=False)
        self.kmer_vectorizer.fit(train_sequences)

        # 3. Calculate raw features (without standardization)
        raw_features = self._transform_sequences(train_sequences, standardize=False)
        self.expected_features = raw_features.shape[1]

        # 4. Fit the scaler
        self.scaler.fit(raw_features)

    def _transform_sequences(self, sequences, standardize=True):
        # Calculate individual features
        features = {
            'aac': np.array(self._parallel_extract(calculate_aac, sequences)),
            'dpc': np.array(self._parallel_extract(calculate_dpc, sequences)),
            'ctd': np.array([calculate_ctd(seq, self.pc_props) for seq in sequences]),
            'complexity': np.array([calculate_sequence_complexity(seq) for seq in sequences])
        }

        # Apply DPC mask
        if hasattr(self, 'dpc_mask'):
            features['dpc'] = features['dpc'][:, self.dpc_mask]
        else:
            raise RuntimeError("DPC mask not initialized! Please run fit() first.")

        # Process k-mer features
        kmer_features = self.kmer_vectorizer.transform(sequences).toarray()
        if kmer_features.shape[1] < 50:
            padding = np.zeros((len(sequences), 50 - kmer_features.shape[1]))
            kmer_features = np.hstack([kmer_features, padding])
        features['kmer'] = kmer_features

        # Combine features
        combined = np.hstack([
            features['aac'], features['dpc'],
            features['ctd'], features['kmer'],
            features['complexity']
        ])

        # Standardize
        if standardize and hasattr(self, 'scaler'):
            combined = self.scaler.transform(combined)
        return combined

    def save_features(self, dataset_name, sequences, labels):
        features = self._transform_sequences(sequences)
        output_path = os.path.join(self.output_dir, f"{dataset_name}_features.npy")
        np.save(output_path, features)
        label_path = os.path.join(self.output_dir, f"{dataset_name}_labels.npy")
        np.save(label_path, labels)
        print(f"✅ Saved {dataset_name} ({len(sequences)} samples) to {output_path}")

def load_fasta(filepath):
    return [str(rec.seq) for rec in SeqIO.parse(filepath, "fasta")]

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Feature Extractor")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Initialize feature extractor
    extractor = ACoVPFeatureExtractor(config)

    # Load datasets
    train_pos = load_fasta(config['data_dir']['train_pos'])
    train_neg = load_fasta(config['data_dir']['train_neg'])
    test_pos = load_fasta(config['data_dir']['test_pos'])
    test_neg = load_fasta(config['data_dir']['test_neg'])
    independent_pos = load_fasta(config['data_dir']['independent_pos'])
    independent_neg = load_fasta(config['data_dir']['independent_neg'])

    # Train feature extractor
    extractor.fit(train_pos + train_neg)

    # Save features
    extractor.save_features("train_pos", train_pos, np.ones(len(train_pos)))
    extractor.save_features("train_neg", train_neg, np.zeros(len(train_neg)))
    extractor.save_features("test_pos", test_pos, np.ones(len(test_pos)))
    extractor.save_features("test_neg", test_neg, np.zeros(len(test_neg)))
    extractor.save_features("independent_pos", independent_pos, np.ones(len(independent_pos)))
    extractor.save_features("independent_neg", independent_neg, np.zeros(len(independent_neg)))

    # Save feature extractor
    extractor_path = os.path.join(extractor.output_dir, "feature_extractor.pkl")
    joblib.dump(extractor, extractor_path)
    print(f"\n🔥 Feature extraction complete! Extractor saved to {extractor_path}")

    # Print feature dimension information
    print(f"""
    ================= Feature Dimension Information =================
    Total features: {extractor.expected_features}
    Breakdown:
      - AAC (Amino Acid Composition): 20
      - DPC (Dipeptide Composition): {extractor.dpc_mask.sum()}
      - CTD (Physicochemical Property Distribution): 98
      - k-mer: 50
      - Sequence Complexity: 7
    ================================================================
    """)