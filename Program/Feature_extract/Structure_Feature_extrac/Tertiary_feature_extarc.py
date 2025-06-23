import os
import numpy as np
import prody
import pickle
import joblib
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import logging
import freesasa
from typing import Dict, Optional, Tuple
from datetime import datetime

class TertiaryStructureFeatureExtractor:
    def __init__(self, output_dir: Optional[str] = None):
        """三级结构特征提取器

        Args:
            output_dir (str): 输出目录路径，如果为None则使用默认路径
        """
        # 先设置输出目录
        self.output_dir = output_dir or '/home/kanglq/code_file/MyProject/Features_model/Features_results/Struc_Feature/Tertiary_Feature'
        os.makedirs(self.output_dir, exist_ok=True)

        # 然后初始化日志系统
        self._setup_logging()

        # 数据集路径配置（更新为新的路径结构）
        self.datasets = {
            'train_pos': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/train_pos',
            'train_neg': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/train_neg',
            'test_pos': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/test_pos',
            'test_neg': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/test_neg',
            'independent_pos': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/independent_pos',
            'independent_neg': '/home/kanglq/code_file/MyProject/Features_model/Data/PDB/independent_neg'
        }

        # 特征参数（将在fit中确定）
        self.contact_thresholds = None
        self.svd_energy_threshold = None
        self.cluster_eps = None
        self.fitted_ = False  # 标记是否已完成拟合

        # 固定参数
        self.min_ca_atoms = 3  # 最少需要3个CA原子
        self.local_contact_window = 3  # 排除局部接触的窗口大小(i±3)

    def _setup_logging(self):
        """配置详细的日志系统"""
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 创建带时间戳的日志文件
        log_file = os.path.join(self.output_dir, f"feature_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("=" * 80)
        self.logger.info("🛠️ 初始化三级结构特征提取器")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info("=" * 80)

    def fit(self):
        """在训练集上拟合特征提取器参数"""
        if self.fitted_:
            self.logger.warning("特征提取器已拟合，跳过重复拟合")
            return self

        self.logger.info("=" * 80)
        self.logger.info("🏋️ 开始训练阶段（仅使用train_pos和train_neg）")

        # 1. 确定最优接触距离阈值
        self._determine_contact_thresholds()

        # 2. 确定SVD能量阈值（基于训练数据）
        self.svd_energy_threshold = 0.9

        # 3. 确定带电簇检测的DBSCAN参数
        self.cluster_eps = 5.0

        self.fitted_ = True
        self._save_extractor()

        self.logger.info("✅ 特征提取器训练完成")
        self.logger.info(f"接触距离阈值: {self.contact_thresholds}")
        self.logger.info(f"SVD能量阈值: {self.svd_energy_threshold}")
        self.logger.info(f"带电簇检测EPS: {self.cluster_eps}")
        self.logger.info("=" * 80)

        return self

    def _determine_contact_thresholds(self):
        """基于训练数据统计确定接触距离阈值"""
        # 这里简化处理，实际应用中应该从训练数据统计得出
        self.contact_thresholds = {
            'strict': 4.0,  # 紧密接触
            'regular': 6.0,  # 常规接触
            'long_range': 8.0  # 长程接触
        }

    def _save_extractor(self):
        """保存训练好的特征提取器"""
        extractor_path = os.path.join(self.output_dir, "feature_extractor.pkl")
        joblib.dump(self, extractor_path)
        self.logger.info(f"💾 保存特征提取器到 {extractor_path}")

    @classmethod
    def load_extractor(cls, path: str):
        """加载已保存的特征提取器"""
        extractor = joblib.load(path)
        extractor.logger.info(f"🔍 从 {path} 加载特征提取器")
        return extractor

    def _get_ca_coords(self, pdb_path: str) -> Optional[np.ndarray]:
        """获取CA原子坐标，带严格检查

        Args:
            pdb_path (str): PDB文件路径

        Returns:
            Optional[np.ndarray]: CA原子坐标数组，如果失败则返回None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(pdb_path):
                self.logger.error(f"❌ PDB文件不存在: {pdb_path}")
                return None

            # 检查文件是否为空
            if os.path.getsize(pdb_path) == 0:
                self.logger.error(f"❌ 空PDB文件: {pdb_path}")
                return None

            # 使用ProDy解析PDB
            structure = prody.parsePDB(pdb_path)
            if structure is None:
                self.logger.error(f"❌ 无法解析PDB文件: {pdb_path}")
                return None

            # 选择CA原子
            calphas = structure.select('name CA')
            if calphas is None or len(calphas) < self.min_ca_atoms:
                self.logger.warning(
                    f"⚠️ CA原子数不足({len(calphas) if calphas else 0} < {self.min_ca_atoms}): {pdb_path}")
                return None

            return calphas.getCoords()

        except Exception as e:
            self.logger.error(f"❌ 解析PDB文件出错 {pdb_path}: {str(e)}", exc_info=True)
            return None

    def _calculate_contact_map(self, coords: np.ndarray, threshold: float) -> np.ndarray:
        """计算接触图，排除局部接触

        Args:
            coords (np.ndarray): CA原子坐标数组
            threshold (float): 接触距离阈值

        Returns:
            np.ndarray: 接触图矩阵
        """
        try:
            # 计算距离矩阵
            dist_matrix = squareform(pdist(coords))

            # 创建接触图
            contact_map = (dist_matrix <= threshold).astype(int)

            # 排除局部接触(i±window)
            n = contact_map.shape[0]
            for i in range(n):
                contact_map[i, max(0, i - self.local_contact_window):min(n, i + self.local_contact_window + 1)] = 0

            # 确保对称性
            return np.maximum(contact_map, contact_map.T)

        except Exception as e:
            self.logger.error(f"❌ 计算接触图出错: {str(e)}", exc_info=True)
            return np.zeros((len(coords), len(coords)))

    def _svd_features(self, matrix: np.ndarray) -> np.ndarray:
        """计算SVD特征，带鲁棒性处理"""
        try:
            if not self.fitted_:
                raise RuntimeError("特征提取器尚未拟合！请先调用fit()方法")

            # 矩阵预处理 - 确保非负且对称
            matrix = np.abs(matrix)  # 确保所有值为非负
            matrix = np.maximum(matrix, matrix.T)  # 确保对称

            # 检查全零矩阵或无效矩阵
            if np.all(matrix == 0) or matrix.shape[0] < 2:
                return np.zeros(5)

            # 归一化处理
            matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-10)

            # 添加微小噪声避免奇异矩阵
            matrix = matrix + 1e-10 * np.random.rand(*matrix.shape)

            # 动态确定组件数量
            n_components = min(matrix.shape[0] - 1, 5)
            if n_components < 1:
                return np.zeros(5)

            # 计算SVD
            svd = TruncatedSVD(n_components=n_components,
                               algorithm='arpack',
                               random_state=42)
            svd.fit(matrix)

            # 提取特征并确保长度为5
            features = svd.transform(matrix)[0, :n_components]  # 取第一行特征
            features = np.pad(features, (0, 5 - len(features)), 'constant')[:5]

            return features

        except Exception as e:
            self.logger.warning(f"⚠️ SVD计算失败: {str(e)}")
            return np.zeros(5)

    def _geometric_features(self, coords: np.ndarray) -> Dict[str, float]:
        """计算几何特征

        Args:
            coords (np.ndarray): CA原子坐标数组

        Returns:
            Dict[str, float]: 几何特征字典
        """
        try:
            # 计算质心
            centroid = np.mean(coords, axis=0)

            # 回转半径
            radius_gyration = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))

            # 首末端距离比
            end_to_end = np.linalg.norm(coords[0] - coords[-1])
            max_theoretical = 3.8 * (len(coords) - 1)
            end_ratio = end_to_end / max_theoretical if max_theoretical > 0 else 0

            return {
                'radius_gyration': float(radius_gyration),
                'end_to_end_ratio': float(end_ratio)
            }

        except Exception as e:
            self.logger.error(f"❌ 计算几何特征出错: {str(e)}", exc_info=True)
            return {
                'radius_gyration': 0.0,
                'end_to_end_ratio': 0.0
            }

    def _calculate_rsa(self, pdb_path: str) -> Dict[str, float]:
        """计算相对溶剂可及表面积(RSA)

        Args:
            pdb_path (str): PDB文件路径

        Returns:
            Dict[str, float]: RSA特征字典
        """
        default_result = {
            'rsa_25th': 0.0,
            'rsa_50th': 0.0,
            'rsa_75th': 0.0
        }

        try:
            # 检查文件有效性
            if not os.path.exists(pdb_path) or os.path.getsize(pdb_path) == 0:
                return default_result

            # 使用FreeSASA计算
            structure = freesasa.Structure(pdb_path)
            result = freesasa.calc(structure)

            # 收集SASA值
            sasa_values = []
            for i in range(structure.nAtoms()):
                try:
                    sasa_values.append(result.atomArea(i))
                except:
                    continue

            # 检查是否有有效数据
            if not sasa_values:
                return default_result

            # 计算百分位数
            return {
                'rsa_25th': float(np.percentile(sasa_values, 25)),
                'rsa_50th': float(np.percentile(sasa_values, 50)),
                'rsa_75th': float(np.percentile(sasa_values, 75))
            }

        except Exception as e:
            self.logger.error(f"❌ RSA计算失败 {pdb_path}: {str(e)}", exc_info=True)
            return default_result

    def _detect_charged_clusters(self, pdb_path: str) -> Dict[str, int]:
        """检测带电残基簇

        Args:
            pdb_path (str): PDB文件路径

        Returns:
            Dict[str, int]: 带电簇特征字典
        """
        try:
            structure = prody.parsePDB(pdb_path)
            if structure is None:
                return {'num_clusters': 0}

            # 选择带电残基(ARG, LYS, ASP, GLU)
            charged_residues = structure.select('resname ARG LYS ASP GLU')
            if charged_residues is None or len(charged_residues) < 2:
                return {'num_clusters': 0}

            # 使用DBSCAN聚类
            coords = charged_residues.getCoords()
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=2).fit(coords)

            # 统计簇数量(排除噪声点)
            labels = clustering.labels_
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            return {'num_clusters': int(num_clusters)}

        except Exception as e:
            self.logger.error(f"❌ 带电簇检测失败 {pdb_path}: {str(e)}", exc_info=True)
            return {'num_clusters': 0}

    def _default_features(self) -> Dict:
        """生成默认特征

        Returns:
            Dict: 默认特征字典
        """
        features = {
            'radius_gyration': 0.0,
            'end_to_end_ratio': 0.0,
            'rsa_25th': 0.0,
            'rsa_50th': 0.0,
            'rsa_75th': 0.0,
            'num_clusters': 0
        }

        # 添加SVD特征的默认值（展开为5个标量）
        if self.contact_thresholds:
            for name in self.contact_thresholds:
                for i in range(5):
                    features[f'svd_{name}_{i}'] = 0.0

        return features

    def extract_features(self, pdb_path: str) -> Dict:
        """从单个PDB文件提取特征

        Args:
            pdb_path (str): PDB文件路径

        Returns:
            Dict: 特征字典
        """
        if not self.fitted_:
            raise RuntimeError("请先调用fit()方法训练特征提取器！")

        self.logger.info(f"🔍 处理文件: {pdb_path}")

        # 获取CA原子坐标
        coords = self._get_ca_coords(pdb_path)
        if coords is None:
            self.logger.warning(f"⚠️ 使用默认特征: {pdb_path}")
            return self._default_features()

        features = {}

        # 1. 几何特征
        features.update(self._geometric_features(coords))

        # 2. 接触图和SVD特征 - 将数组特征展开为多个标量特征
        for name, threshold in self.contact_thresholds.items():
            cmap = self._calculate_contact_map(coords, threshold)
            svd_features = self._svd_features(cmap)
            # 将5维SVD特征展开为5个单独的特征
            for i in range(5):
                features[f'svd_{name}_{i}'] = float(svd_features[i])

        # 3. RSA特征
        features.update(self._calculate_rsa(pdb_path))

        # 4. 带电簇特征
        features.update(self._detect_charged_clusters(pdb_path))

        self.logger.debug(f"📊 提取的特征: {features}")
        return features

    def process_dataset(self, dataset_name: str) -> Dict[str, Dict]:
        """处理整个数据集

        Args:
            dataset_name (str): 数据集名称(train_pos/train_neg/test_pos/test_neg/independent_pos/independent_neg)

        Returns:
            Dict[str, Dict]: 特征字典{肽段ID: 特征}
        """
        if not self.fitted_ and dataset_name.startswith(('test', 'independent')):
            raise RuntimeError("请先调用fit()方法训练特征提取器！")

        self.logger.info("=" * 80)
        self.logger.info(f"📂 开始处理数据集: {dataset_name}")

        input_dir = self.datasets[dataset_name]
        features = []
        labels = []

        # 获取PDB文件列表
        pdb_files = [f for f in os.listdir(input_dir) if f.endswith('.pdb')]

        # 获取特征顺序（使用默认特征作为模板）
        feature_order = list(self._default_features().keys())

        # 使用进度条处理文件
        for filename in tqdm(pdb_files,
                             desc=f"处理 {dataset_name}"):
            pdb_path = os.path.join(input_dir, filename)
            peptide_id = os.path.splitext(filename)[0]

            try:
                feat = self.extract_features(pdb_path)
                if feat is not None:
                    # 确保特征顺序一致
                    ordered_feat = [feat[key] for key in feature_order]
                    features.append(ordered_feat)
                    labels.append(1 if 'pos' in dataset_name else 0)
            except Exception as e:
                self.logger.error(f"❌ 处理 {filename} 出错: {str(e)}", exc_info=True)

        # 转换为NumPy数组
        if features:
            feature_array = np.array(features, dtype=np.float32)
        else:
            feature_array = np.zeros((0, len(feature_order)), dtype=np.float32)

        label_array = np.array(labels, dtype=np.int32)

        # 保存特征和标签
        feature_path = os.path.join(self.output_dir, f"{dataset_name}_features.npy")
        label_path = os.path.join(self.output_dir, f"{dataset_name}_labels.npy")
        np.save(feature_path, feature_array)
        np.save(label_path, label_array)

        self.logger.info(f"💾 保存 {dataset_name} 特征到 {feature_path}")
        self.logger.info(f"  - 特征矩阵形状: {feature_array.shape}")
        self.logger.info(f"  - 标签矩阵形状: {label_array.shape}")

        self.logger.info("=" * 80)
        return feature_array

    def process_all_datasets(self) -> Dict[str, Dict[str, Dict]]:
        """执行完整特征提取流程

        Returns:
            Dict[str, Dict[str, Dict]]: 所有数据集的特征
        """
        self.logger.info("=" * 80)
        self.logger.info("🚀 开始三级结构特征提取流程")

        # 1. 训练阶段（仅使用训练数据）
        self.fit()

        # 2. 处理所有数据集（包括独立数据集）
        all_features = {}
        for dataset in ['train_pos', 'train_neg', 'test_pos', 'test_neg', 'independent_pos', 'independent_neg']:
            try:
                all_features[dataset] = self.process_dataset(dataset)
            except Exception as e:
                self.logger.error(f"❌ 处理数据集 {dataset} 失败: {str(e)}", exc_info=True)

        self.logger.info("🎉 特征提取完成！")
        self.logger.info("=" * 80)
        return all_features


if __name__ == "__main__":
    # 初始化特征提取器
    extractor = TertiaryStructureFeatureExtractor()

    # 执行完整流程
    all_features = extractor.process_all_datasets()