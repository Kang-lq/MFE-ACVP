import os
import numpy as np
import prody
import joblib
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import logging
import freesasa
from typing import Dict, Optional, Tuple
from datetime import datetime
import argparse
import yaml

class TertiaryStructureFeatureExtractor:
    def __init__(self, config: Dict):
        """
        Tertiary structure feature extractor for PDB files.

        Args:
            config (Dict): Configuration dictionary containing output directory and dataset paths.
        """
        # Set output directory
        self.output_dir = config.get('output_dir', './output')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize logging system
        self._setup_logging()

        # Dataset path configuration
        self.datasets = {
            'train_pos': config['data_root']['train_pos'],
            'train_neg': config['data_root']['train_neg'],
            'test_pos': config['data_root']['test_pos'],
            'test_neg': config['data_root']['test_neg'],
            'independent_pos': config['data_root']['independent_pos'],
            'independent_neg': config['data_root']['independent_neg']
        }

        # Feature parameters (to be determined in fit)
        self.contact_thresholds = None
        self.svd_energy_threshold = None
        self.cluster_eps = None
        self.fitted_ = False  # Flag to indicate if fitting is completed

        # Fixed parameters
        self.min_ca_atoms = 3  # Minimum number of CA atoms required
        self.local_contact_window = 3  # Window size for excluding local contacts (i±3)

    def _setup_logging(self):
        """
        Configure the detailed logging system.
        """
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Create a log file with timestamp
        log_file = os.path.join(self.output_dir, f"feature_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("=" * 80)
        self.logger.info("🛠️ Initializing tertiary structure feature extractor")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 80)

    def fit(self):
        """
        Fit the feature extractor parameters on the training set.
        """
        if self.fitted_:
            self.logger.warning("Feature extractor already fitted, skipping repeated fitting")
            return self

        self.logger.info("=" * 80)
        self.logger.info("🏋️ Starting training phase (using train_pos and train_neg only)")

        # 1. Determine optimal contact distance thresholds
        self._determine_contact_thresholds()

        # 2. Determine SVD energy threshold (based on training data)
        self.svd_energy_threshold = 0.9

        # 3. Determine DBSCAN parameters for charged cluster detection
        self.cluster_eps = 5.0

        self.fitted_ = True
        self._save_extractor()

        self.logger.info("✅ Feature extractor training completed")
        self.logger.info(f"Contact distance thresholds: {self.contact_thresholds}")
        self.logger.info(f"SVD energy threshold: {self.svd_energy_threshold}")
        self.logger.info(f"Charged cluster detection EPS: {self.cluster_eps}")
        self.logger.info("=" * 80)

        return self

    def _determine_contact_thresholds(self):
        """
        Determine contact distance thresholds based on training data statistics.
        """
        # Simplified handling; in practice, this should be derived from training data statistics
        self.contact_thresholds = {
            'strict': 4.0,  # Tight contact
            'regular': 6.0,  # Regular contact
            'long_range': 8.0  # Long-range contact
        }

    def _save_extractor(self):
        """
        Save the trained feature extractor.
        """
        extractor_path = os.path.join(self.output_dir, "feature_extractor.pkl")
        joblib.dump(self, extractor_path)
        self.logger.info(f"💾 Saving feature extractor to {extractor_path}")

    @classmethod
    def load_extractor(cls, path: str):
        """
        Load a saved feature extractor.

        Args:
            path (str): Path to the saved feature extractor file.

        Returns:
            TertiaryStructureFeatureExtractor: Loaded feature extractor.
        """
        extractor = joblib.load(path)
        extractor.logger.info(f"🔍 Loading feature extractor from {path}")
        return extractor

    def _get_ca_coords(self, pdb_path: str) -> Optional[np.ndarray]:
        """
        Get CA atom coordinates with strict checks.

        Args:
            pdb_path (str): Path to the PDB file.

        Returns:
            Optional[np.ndarray]: CA atom coordinates array, or None if failed.
        """
        try:
            # Check if the file exists
            if not os.path.exists(pdb_path):
                self.logger.error(f"❌ PDB file does not exist: {pdb_path}")
                return None

            # Check if the file is empty
            if os.path.getsize(pdb_path) == 0:
                self.logger.error(f"❌ Empty PDB file: {pdb_path}")
                return None

            # Parse PDB using ProDy
            structure = prody.parsePDB(pdb_path)
            if structure is None:
                self.logger.error(f"❌ Failed to parse PDB file: {pdb_path}")
                return None

            # Select CA atoms
            calphas = structure.select('name CA')
            if calphas is None or len(calphas) < self.min_ca_atoms:
                self.logger.warning(
                    f"⚠️ Insufficient CA atoms ({len(calphas) if calphas else 0} < {self.min_ca_atoms}): {pdb_path}")
                return None

            return calphas.getCoords()

        except Exception as e:
            self.logger.error(f"❌ Error parsing PDB file {pdb_path}: {str(e)}", exc_info=True)
            return None

    def _calculate_contact_map(self, coords: np.ndarray, threshold: float) -> np.ndarray:
        """
        Calculate the contact map, excluding local contacts.

        Args:
            coords (np.ndarray): CA atom coordinates array.
            threshold (float): Contact distance threshold.

        Returns:
            np.ndarray: Contact map matrix.
        """
        try:
            # Calculate distance matrix
            dist_matrix = squareform(pdist(coords))

            # Create contact map
            contact_map = (dist_matrix <= threshold).astype(int)

            # Exclude local contacts (i±window)
            n = contact_map.shape[0]
            for i in range(n):
                contact_map[i, max(0, i - self.local_contact_window):min(n, i + self.local_contact_window + 1)] = 0

            # Ensure symmetry
            return np.maximum(contact_map, contact_map.T)

        except Exception as e:
            self.logger.error(f"❌ Error calculating contact map: {str(e)}", exc_info=True)
            return np.zeros((len(coords), len(coords)))

    def _svd_features(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate SVD features with robust handling.

        Args:
            matrix (np.ndarray): Input matrix.

        Returns:
            np.ndarray: SVD features array.
        """
        try:
            if not self.fitted_:
                raise RuntimeError("Feature extractor not fitted! Please call fit() method first")

            # Matrix preprocessing - ensure non-negative and symmetric
            matrix = np.abs(matrix)  # Ensure all values are non-negative
            matrix = np.maximum(matrix, matrix.T)  # Ensure symmetry

            # Check for zero matrix or invalid matrix
            if np.all(matrix == 0) or matrix.shape[0] < 2:
                return np.zeros(5)

            # Normalize the matrix
            matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-10)

            # Add a small noise to avoid singular matrix
            matrix = matrix + 1e-10 * np.random.rand(*matrix.shape)

            # Dynamically determine the number of components
            n_components = min(matrix.shape[0] - 1, 5)
            if n_components < 1:
                return np.zeros(5)

            # Calculate SVD
            svd = TruncatedSVD(n_components=n_components,
                               algorithm='arpack',
                               random_state=42)
            svd.fit(matrix)

            # Extract features and ensure length is 5
            features = svd.transform(matrix)[0, :n_components]  # Take the first row of features
            features = np.pad(features, (0, 5 - len(features)), 'constant')[:5]

            return features

        except Exception as e:
            self.logger.warning(f"⚠️ SVD calculation failed: {str(e)}")
            return np.zeros(5)

    def _geometric_features(self, coords: np.ndarray) -> Dict[str, float]:
        """
        Calculate geometric features.

        Args:
            coords (np.ndarray): CA atom coordinates array.

        Returns:
            Dict[str, float]: Dictionary of geometric features.
        """
        try:
            # Calculate centroid
            centroid = np.mean(coords, axis=0)

            # Radius of gyration
            radius_gyration = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))

            # End-to-end distance ratio
            end_to_end = np.linalg.norm(coords[0] - coords[-1])
            max_theoretical = 3.8 * (len(coords) - 1)
            end_ratio = end_to_end / max_theoretical if max_theoretical > 0 else 0

            return {
                'radius_gyration': float(radius_gyration),
                'end_to_end_ratio': float(end_ratio)
            }

        except Exception as e:
            self.logger.error(f"❌ Error calculating geometric features: {str(e)}", exc_info=True)
            return {
                'radius_gyration': 0.0,
                'end_to_end_ratio': 0.0
            }

    def _calculate_rsa(self, pdb_path: str) -> Dict[str, float]:
        """
        Calculate relative solvent accessible surface area (RSA).

        Args:
            pdb_path (str): Path to the PDB file.

        Returns:
            Dict[str, float]: Dictionary of RSA features.
        """
        default_result = {
            'rsa_25th': 0.0,
            'rsa_50th': 0.0,
            'rsa_75th': 0.0
        }

        try:
            # Check file validity
            if not os.path.exists(pdb_path) or os.path.getsize(pdb_path) == 0:
                return default_result

            # Calculate using FreeSASA
            structure = freesasa.Structure(pdb_path)
            result = freesasa.calc(structure)

            # Collect SASA values
            sasa_values = []
            for i in range(structure.nAtoms()):
                try:
                    sasa_values.append(result.atomArea(i))
                except:
                    continue

            # Check for valid data
            if not sasa_values:
                return default_result

            # Calculate percentiles
            return {
                'rsa_25th': float(np.percentile(sasa_values, 25)),
                'rsa_50th': float(np.percentile(sasa_values, 50)),
                'rsa_75th': float(np.percentile(sasa_values, 75))
            }

        except Exception as e:
            self.logger.error(f"❌ RSA calculation failed {pdb_path}: {str(e)}", exc_info=True)
            return default_result

    def _detect_charged_clusters(self, pdb_path: str) -> Dict[str, int]:
        """
        Detect charged residue clusters.

        Args:
            pdb_path (str): Path to the PDB file.

        Returns:
            Dict[str, int]: Dictionary of charged cluster features.
        """
        try:
            structure = prody.parsePDB(pdb_path)
            if structure is None:
                return {'num_clusters': 0}

            # Select charged residues (ARG, LYS, ASP, GLU)
            charged_residues = structure.select('resname ARG LYS ASP GLU')
            if charged_residues is None or len(charged_residues) < 2:
                return {'num_clusters': 0}

            # Perform DBSCAN clustering
            coords = charged_residues.getCoords()
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=2).fit(coords)

            # Count clusters (excluding noise points)
            labels = clustering.labels_
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            return {'num_clusters': int(num_clusters)}

        except Exception as e:
            self.logger.error(f"❌ Charged cluster detection failed {pdb_path}: {str(e)}", exc_info=True)
            return {'num_clusters': 0}

    def _default_features(self) -> Dict:
        """
        Generate default features.

        Returns:
            Dict: Dictionary of default features.
        """
        features = {
            'radius_gyration': 0.0,
            'end_to_end_ratio': 0.0,
            'rsa_25th': 0.0,
            'rsa_50th': 0.0,
            'rsa_75th': 0.0,
            'num_clusters': 0
        }

        # Add default values for SVD features (expanded to 5 scalars)
        if self.contact_thresholds:
            for name in self.contact_thresholds:
                for i in range(5):
                    features[f'svd_{name}_{i}'] = 0.0

        return features

    def extract_features(self, pdb_path: str) -> Dict:
        """
        Extract features from a single PDB file.

        Args:
            pdb_path (str): Path to the PDB file.

        Returns:
            Dict: Dictionary of features.
        """
        if not self.fitted_:
            raise RuntimeError("Please call fit() method to train the feature extractor first!")

        self.logger.info(f"🔍 Processing file: {pdb_path}")

        # Get CA atom coordinates
        coords = self._get_ca_coords(pdb_path)
        if coords is None:
            self.logger.warning(f"⚠️ Using default features: {pdb_path}")
            return self._default_features()

        features = {}

        # 1. Geometric features
        features.update(self._geometric_features(coords))

        # 2. Contact map and SVD features - expand array features into multiple scalar features
        for name, threshold in self.contact_thresholds.items():
            cmap = self._calculate_contact_map(coords, threshold)
            svd_features = self._svd_features(cmap)
            # Expand 5-dimensional SVD features into 5 separate features
            for i in range(5):
                features[f'svd_{name}_{i}'] = float(svd_features[i])

        # 3. RSA features
        features.update(self._calculate_rsa(pdb_path))

        # 4. Charged cluster features
        features.update(self._detect_charged_clusters(pdb_path))

        self.logger.debug(f"📊 Extracted features: {features}")
        return features

    def process_dataset(self, dataset_name: str) -> Dict[str, Dict]:
        """
        Process an entire dataset.

        Args:
            dataset_name (str): Dataset name (train_pos/train_neg/test_pos/test_neg/independent_pos/independent_neg)

        Returns:
            Dict[str, Dict]: Dictionary of features {peptide ID: features}
        """
        if not self.fitted_ and dataset_name.startswith(('test', 'independent')):
            raise RuntimeError("Please call fit() method to train the feature extractor first!")

        self.logger.info("=" * 80)
        self.logger.info(f"📂 Starting to process dataset: {dataset_name}")

        input_dir = self.datasets[dataset_name]
        features = []
        labels = []

        # Get list of PDB files
        pdb_files = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]

        # Get feature order (using default features as a template)
        feature_order = list(self._default_features().keys())

        # Use progress bar to process files
        for filename in tqdm(pdb_files, desc=f"Processing {dataset_name}"):
            pdb_path = os.path.join(input_dir, filename)
            peptide_id = os.path.splitext(filename)[0]

            try:
                feat = self.extract_features(pdb_path)
                if feat is not None:
                    # Ensure feature order is consistent
                    ordered_feat = [feat[key] for key in feature_order]
                    features.append(ordered_feat)
                    labels.append(1 if 'pos' in dataset_name else 0)
            except Exception as e:
                self.logger.error(f"❌ Error processing {filename}: {str(e)}", exc_info=True)

        # Convert to NumPy arrays
        if features:
            feature_array = np.array(features, dtype=np.float32)
        else:
            feature_array = np.zeros((0, len(feature_order)), dtype=np.float32)

        label_array = np.array(labels, dtype=np.int32)

        # Save features and labels
        feature_path = os.path.join(self.output_dir, f"{dataset_name}_features.npy")
        label_path = os.path.join(self.output_dir, f"{dataset_name}_labels.npy")
        np.save(feature_path, feature_array)
        np.save(label_path, label_array)

        self.logger.info(f"💾 Saving {dataset_name} features to {feature_path}")
        self.logger.info(f"  - Feature matrix shape: {feature_array.shape}")
        self.logger.info(f"  - Label matrix shape: {label_array.shape}")

        self.logger.info("=" * 80)
        return feature_array

    def process_all_datasets(self) -> Dict[str, Dict[str, Dict]]:
        """
        Execute the complete feature extraction workflow.

        Returns:
            Dict[str, Dict[str, Dict]]: Features for all datasets.
        """
        self.logger.info("=" * 80)
        self.logger.info("🚀 Starting tertiary structure feature extraction workflow")

        # 1. Training phase (using training data only)
        self.fit()

        # 2. Process all datasets (including independent datasets)
        all_features = {}
        for dataset in ['train_pos', 'train_neg', 'test_pos', 'test_neg', 'independent_pos', 'independent_neg']:
            try:
                all_features[dataset] = self.process_dataset(dataset)
            except Exception as e:
                self.logger.error(f"❌ Failed to process dataset {dataset}: {str(e)}", exc_info=True)

        self.logger.info("🎉 Feature extraction completed!")
        self.logger.info("=" * 80)
        return all_features


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Tertiary Structure Feature Extractor")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Initialize feature extractor
    extractor = TertiaryStructureFeatureExtractor(config)

    # Execute the complete workflow
    all_features = extractor.process_all_datasets()