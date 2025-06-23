"""
Graph-based structural feature extractor for peptide PDB files
Modified for clean GitHub release (removes hard-coded paths and GPU ID)
"""

import os
import numpy as np
import prody
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE, global_mean_pool
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import joblib
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import networkx as nx
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore", module="prody")
prody.confProDy(verbosity='none')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class GraphEmbedder(nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.conv1 = GraphSAGE(
            in_channels=21,
            hidden_channels=hidden_channels,
            num_layers=2,
            out_channels=hidden_channels
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        return global_mean_pool(x, batch)


class GraphFeatureExtractor:
    def __init__(self, data_root: str, output_dir: str, gpu_ids: Optional[List[int]] = None, embed_dim: int = 64):
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
        self._setup_logging()

        self.aa_to_idx = {
            'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
            'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
            'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
            'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19
        }
        self.spatial_cutoff = None
        self.fitted_ = False
        self.embed_dim = embed_dim

        self.gnn = GraphEmbedder(hidden_channels=embed_dim)
        if gpu_ids and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_ids[0]}')
            self.gnn = self.gnn.to(self.device)
        else:
            self.device = torch.device('cpu')

        self.logger.info(f"Using device: {self.device} | Embedding dim: {embed_dim}")

    def _setup_logging(self):
        self.logger = logging.getLogger('GraphFeatureExtractor')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        log_file = os.path.join(
            self.output_dir,
            f"graph_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _construct_graph(self, pdb_path: str) -> Optional[Data]:
        try:
            structure = prody.parsePDB(pdb_path)
            if structure is None:
                return None
            heavy_atoms = structure.select('heavy')
            if heavy_atoms is None or len(heavy_atoms) < 3:
                return None

            node_features = []
            atom_indices = {}
            valid_atoms = 0
            for atom in heavy_atoms:
                aa_idx = self.aa_to_idx.get(atom.getResname(), -1)
                if aa_idx >= 0:
                    aa_onehot = np.zeros(20, dtype=np.float32)
                    aa_onehot[aa_idx] = 1
                    sasa_norm = min((atom.getBeta() or 0.0) / 200.0, 1.0)
                    node_features.append(np.concatenate([aa_onehot, [sasa_norm]]))
                    atom_indices[atom.getIndex()] = valid_atoms
                    valid_atoms += 1

            if valid_atoms < 3:
                return None

            node_tensor = torch.from_numpy(np.array(node_features, dtype=np.float32))
            coords = heavy_atoms.getCoords()
            dist_matrix = squareform(pdist(coords))
            edge_index = []

            prev_atom = None
            for atom in heavy_atoms:
                if (prev_atom and atom.getResnum() == prev_atom.getResnum() + 1 and
                        prev_atom.getName() == 'C' and atom.getName() == 'N' and
                        prev_atom.getIndex() in atom_indices and atom.getIndex() in atom_indices):
                    i, j = atom_indices[prev_atom.getIndex()], atom_indices[atom.getIndex()]
                    edge_index.extend([[i, j], [j, i]])
                prev_atom = atom

            cutoff = self.spatial_cutoff or 8.0
            for i in range(valid_atoms):
                for j in range(i + 1, valid_atoms):
                    if dist_matrix[i, j] < cutoff:
                        edge_index.extend([[i, j], [j, i]])

            if not edge_index:
                return None

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            return Data(
                x=node_tensor,
                edge_index=edge_index,
                edge_attr=torch.tensor(dist_matrix[edge_index[0], edge_index[1]], dtype=torch.float32).unsqueeze(1),
                num_nodes=valid_atoms,
                batch=torch.zeros(valid_atoms, dtype=torch.long),
                pdb_path=pdb_path
            )
        except Exception as e:
            self.logger.error(f"Failed to build graph for {pdb_path}: {str(e)}")
            return None

    def _extract_features(self, graph: Data) -> Dict:
        if graph is None:
            return {'embedding': np.zeros(self.embed_dim), 'stats': np.zeros(5), 'valid': False}

        try:
            graph = graph.to(self.device)
            with torch.no_grad():
                embedding = self.gnn(graph).cpu().numpy().flatten()

            adj = csr_matrix((np.ones(graph.edge_index.size(1)), graph.edge_index.cpu().numpy()),
                             shape=(graph.num_nodes, graph.num_nodes))
            G = nx.from_scipy_sparse_array(adj)
            degrees = dict(G.degree()).values()
            avg_degree = sum(degrees) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
            clust_coef = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
            n_components = nx.number_connected_components(G)

            stats = np.array([avg_degree, clust_coef, n_components, graph.num_nodes,
                              graph.edge_index.size(1) // 2], dtype=np.float32)

            return {'embedding': embedding, 'stats': stats, 'valid': True}
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            return {'embedding': np.zeros(self.embed_dim), 'stats': np.zeros(5), 'valid': False}

    def _process_single_file(self, dataset: str, filename: str) -> Tuple[str, Optional[Dict]]:
        pdb_path = os.path.join(self.datasets[dataset], filename)
        graph = self._construct_graph(pdb_path)
        features = self._extract_features(graph)
        if features['valid']:
            return os.path.splitext(filename)[0], np.concatenate([features['embedding'], features['stats']])
        else:
            return os.path.splitext(filename)[0], None

    def process_dataset(self, dataset: str, n_jobs: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        pdb_files = [f for f in os.listdir(self.datasets[dataset]) if f.endswith('.pdb')]
        self.logger.info(f"Processing {dataset} ({len(pdb_files)} files)...")

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_single_file)(dataset, f)
            for f in tqdm(pdb_files, desc=f"{dataset}")
        )

        valid_results = [r for r in results if r[1] is not None]
        features = np.array([r[1] for r in valid_results])
        labels = np.array([1 if 'pos' in dataset else 0 for _ in valid_results], dtype=np.int32)

        self.logger.info(f"Done {dataset}: {len(features)}/{len(pdb_files)} valid")
        return features, labels

    def fit(self):
        if self.fitted_:
            return self
        self.logger.info("Calculating spatial distance cutoff...")
        all_distances = []
        for dataset in ['train_pos', 'train_neg']:
            for filename in os.listdir(self.datasets[dataset]):
                if not filename.endswith('.pdb'):
                    continue
                try:
                    coords = prody.parsePDB(os.path.join(self.datasets[dataset], filename)).select('heavy').getCoords()
                    dist_matrix = squareform(pdist(coords))
                    np.fill_diagonal(dist_matrix, np.inf)
                    all_distances.extend(dist_matrix[dist_matrix < 20].flatten())
                except:
                    continue
        self.spatial_cutoff = np.percentile(all_distances, 75) if all_distances else 8.0
        if self.spatial_cutoff > 15:
            self.logger.warning(f"Cutoff too high: {self.spatial_cutoff:.2f}, resetting to 10")
            self.spatial_cutoff = 10.0
        self.fitted_ = True
        self.logger.info(f"Cutoff set to {self.spatial_cutoff:.2f} Å")
        return self

    def _save_state(self):
        state = {
            'spatial_cutoff': self.spatial_cutoff,
            'fitted_': self.fitted_,
            'embed_dim': self.embed_dim,
            'aa_to_idx': self.aa_to_idx,
            'gnn': self.gnn.state_dict()
        }
        joblib.dump(state, os.path.join(self.output_dir, "extractor_state.pkl"))

    def process_all_datasets(self, n_jobs: int = 4):
        self.fit()
        for dataset in self.datasets:
            features, labels = self.process_dataset(dataset, n_jobs=n_jobs)
            np.save(os.path.join(self.output_dir, f"{dataset}_features.npy"), features)
            np.save(os.path.join(self.output_dir, f"{dataset}_labels.npy"), labels)
        self._save_state()
        self.logger.info("All datasets processed and state saved.")
