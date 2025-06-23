import os
import numpy as np
import prody
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import os

# 设置使用第 2 个 GPU（索引从 0 开始）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 然后在您的设备选择代码中保持原样
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的设备: {device}")
# 关闭ProDy的警告
warnings.filterwarnings("ignore", module="prody")
prody.confProDy(verbosity='none')


class GraphEmbedder(nn.Module):
    """独立的图嵌入模型类（必须定义在模块顶层）"""

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
    def __init__(self, output_dir: str, gpu_ids: List[int] = None, embed_dim: int = 64):
        """完整的图特征提取器

        Args:
            output_dir: 输出目录路径
            gpu_ids: 可用的GPU设备ID列表
            embed_dim: 图嵌入维度
        """
        self.datasets = {
            'train_pos': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/train_pos',
            'train_neg': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/train_neg',
            'test_pos': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/test_pos',
            'test_neg': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/test_neg',
            'independent_pos': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/independent_pos',
            'independent_neg': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/independent_neg'
        }
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()

        # 图构建参数
        self.aa_to_idx = {
            'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
            'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
            'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
            'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19
        }
        self.spatial_cutoff = None
        self.fitted_ = False
        self.embed_dim = embed_dim

        # 初始化GNN模型
        self.gnn = GraphEmbedder(hidden_channels=embed_dim)
        if gpu_ids and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_ids[0]}')
            self.gnn = self.gnn.to(self.device)
        else:
            self.device = torch.device('cpu')

        self.logger.info(f"使用设备: {self.device} | 嵌入维度: {embed_dim}")

    def _setup_logging(self):
        """配置日志系统"""
        self.logger = logging.getLogger('GraphFeatureExtractor')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 文件日志
        log_file = os.path.join(
            self.output_dir,
            f"graph_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _construct_graph(self, pdb_path: str) -> Optional[Data]:
        """构建分子图（已优化张量转换）"""
        try:
            structure = prody.parsePDB(pdb_path)
            if structure is None:
                return None

            heavy_atoms = structure.select('heavy')
            if heavy_atoms is None or len(heavy_atoms) < 3:
                return None

            # 节点特征处理（优化后的高效实现）
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

            # 转换为单一numpy数组再转tensor（解决警告）
            node_features = np.array(node_features, dtype=np.float32)
            node_tensor = torch.from_numpy(node_features)

            # 边索引构建
            coords = heavy_atoms.getCoords()
            dist_matrix = squareform(pdist(coords))
            edge_index = []

            # 肽键连接
            prev_atom = None
            for atom in heavy_atoms:
                if (prev_atom and
                        atom.getResnum() == prev_atom.getResnum() + 1 and
                        prev_atom.getName() == 'C' and atom.getName() == 'N' and
                        prev_atom.getIndex() in atom_indices and
                        atom.getIndex() in atom_indices):
                    i, j = atom_indices[prev_atom.getIndex()], atom_indices[atom.getIndex()]
                    edge_index.extend([[i, j], [j, i]])
                prev_atom = atom

            # 空间邻近连接
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
                edge_attr=torch.tensor(
                    dist_matrix[edge_index[0], edge_index[1]],
                    dtype=torch.float32
                ).unsqueeze(1),
                num_nodes=valid_atoms,
                batch=torch.zeros(valid_atoms, dtype=torch.long),
                pdb_path=pdb_path
            )
        except Exception as e:
            self.logger.error(f"图构建失败 {pdb_path}: {str(e)}")
            return None

    def _extract_features(self, graph: Data) -> Dict:
        """提取图特征（GNN嵌入 + 传统统计）"""
        if graph is None:
            return {
                'embedding': np.zeros(self.embed_dim, dtype=np.float32),
                'stats': np.zeros(5, dtype=np.float32),
                'valid': False
            }

        try:
            # GNN嵌入
            graph = graph.to(self.device)
            with torch.no_grad():
                embedding = self.gnn(graph).cpu().numpy().flatten()

            # 传统统计特征
            adj = csr_matrix(
                (np.ones(graph.edge_index.size(1)),
                 graph.edge_index.cpu().numpy()),
                shape=(graph.num_nodes, graph.num_nodes)
            )
            G = nx.from_scipy_sparse_array(adj)
            degrees = dict(G.degree()).values()
            avg_degree = sum(degrees) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
            clust_coef = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
            n_components = nx.number_connected_components(G)

            stats = np.array([
                avg_degree,
                clust_coef,
                n_components,
                graph.num_nodes,
                graph.edge_index.size(1) // 2
            ], dtype=np.float32)

            return {
                'embedding': embedding,
                'stats': stats,
                'valid': True
            }
        except Exception as e:
            self.logger.error(f"特征提取失败: {str(e)}")
            return {
                'embedding': np.zeros(self.embed_dim, dtype=np.float32),
                'stats': np.zeros(5, dtype=np.float32),
                'valid': False
            }

    def _process_single_file(self, dataset: str, filename: str) -> Tuple[str, Optional[Dict]]:
        """处理单个PDB文件"""
        try:
            pdb_path = os.path.join(self.datasets[dataset], filename)
            graph = self._construct_graph(pdb_path)
            features = self._extract_features(graph)
            if features['valid']:
                combined_features = np.concatenate([features['embedding'], features['stats']])
                return (os.path.splitext(filename)[0], combined_features)
            else:
                return (os.path.splitext(filename)[0], None)
        except Exception as e:
            self.logger.error(f"处理 {filename} 失败: {str(e)}")
            return (os.path.splitext(filename)[0], None)

    def process_dataset(self, dataset: str, n_jobs: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """处理整个数据集"""
        pdb_files = [
            f for f in os.listdir(self.datasets[dataset])
            if f.endswith('.pdb')
        ]

        self.logger.info(f"开始处理 {dataset} ({len(pdb_files)} 个文件)...")

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_single_file)(dataset, f)
            for f in tqdm(pdb_files, desc=f"Processing {dataset}")
        )

        # 合并有效特征
        valid_results = [r for r in results if r[1] is not None]
        features = np.array([r[1] for r in valid_results], dtype=np.float32)

        labels = np.array([
            1 if 'pos' in dataset else 0
            for r in valid_results
        ], dtype=np.int32)

        self.logger.info(
            f"完成 {dataset} | 有效样本: {len(features)}/{len(pdb_files)} | "
            f"特征维度: {features.shape[1]}"
        )
        return features, labels

    def fit(self):
        """确定空间距离阈值"""
        if self.fitted_:
            return self

        self.logger.info("正在计算空间距离阈值...")
        all_distances = []

        # 仅使用训练集计算阈值
        for dataset in ['train_pos', 'train_neg']:
            for filename in os.listdir(self.datasets[dataset]):
                if not filename.endswith('.pdb'):
                    continue

                pdb_path = os.path.join(self.datasets[dataset], filename)
                try:
                    structure = prody.parsePDB(pdb_path)
                    heavy_atoms = structure.select('heavy') if structure else None
                    if heavy_atoms is None:
                        continue

                    coords = heavy_atoms.getCoords()
                    dist_matrix = squareform(pdist(coords))
                    np.fill_diagonal(dist_matrix, np.inf)
                    all_distances.extend(dist_matrix[dist_matrix < 20].flatten())
                except:
                    continue

        self.spatial_cutoff = np.percentile(all_distances, 75) if all_distances else 8.0
        if self.spatial_cutoff > 15.0:
            self.logger.warning(f"空间阈值{self.spatial_cutoff:.2f}Å过高，自动调整为10Å")
            self.spatial_cutoff = 10.0

        self.fitted_ = True
        self.logger.info(f"空间距离阈值设置为: {self.spatial_cutoff:.2f} Å")
        return self

    def _save_state(self):
        """保存可序列化的状态"""
        state = {
            'spatial_cutoff': self.spatial_cutoff,
            'fitted_': self.fitted_,
            'embed_dim': self.embed_dim,
            'aa_to_idx': self.aa_to_idx,
            'gnn': self.gnn.state_dict()  # 保存模型的状态字典
        }
        joblib.dump(state, os.path.join(self.output_dir, "extractor_state.pkl"))

    def process_all_datasets(self, n_jobs: int = 4):
        """处理所有数据集"""
        self.fit()

        # 处理所有数据集（包括独立数据集）
        for dataset in ['train_pos', 'train_neg', 'test_pos', 'test_neg', 'independent_pos', 'independent_neg']:
            features, labels = self.process_dataset(dataset, n_jobs=n_jobs)

            # 保存特征和标签
            np.save(
                os.path.join(self.output_dir, f"{dataset}_features.npy"),
                features
            )
            np.save(
                os.path.join(self.output_dir, f"{dataset}_labels.npy"),
                labels
            )

        # 保存状态（不包含模型）
        self._save_state()
        self.logger.info("所有数据集处理完成，状态已保存")


if __name__ == "__main__":
    # 使用示例
    extractor = GraphFeatureExtractor(
        output_dir="/home/kanglq/code_file/MyProject/Features_model/Features_results/Struc_Feature/Graph_Feature",
        gpu_ids=[0],  # 使用第一个GPU
        embed_dim=64
    )
    extractor.process_all_datasets(n_jobs=4)