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


class TopologicalFeatureExtractor:
    def __init__(self, output_dir: str, log_level: str = "INFO"):
        """
        最终修正的拓扑特征提取器，精确标签分配

        Args:
            output_dir: 特征保存路径
            log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 配置日志系统
        self.logger = logging.getLogger("TopologicalFeatureExtractor")
        self.logger.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 文件日志
        log_file = os.path.join(output_dir, f"topo_feature_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 定义可pickle的描述符计算函数
        def _desc_molwt(mol): return Descriptors.MolWt(mol)

        def _desc_amide(mol): return rdMolDescriptors.CalcNumAmideBonds(mol)

        def _desc_kappa2(mol): return rdMolDescriptors.CalcKappa2(mol)

        def _desc_hdonors(mol): return Lipinski.NumHDonors(mol)

        def _desc_rotatable(mol): return Lipinski.NumRotatableBonds(mol)

        def _desc_aromatic(mol): return Lipinski.NumAromaticRings(mol)

        # 特征配置
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
            "feature_order": None  # 将在首次运行时确定
        }

        # 精确的标识符定义
        self.positive_prefixes = [">ACVP_", ">IACVP_"]  # 正样本必须以此开头
        self.negative_prefixes = [">nACVP_", ">InACVP_"]  # 负样本必须以此开头

        # 特征选择器和标准化器
        self.selector = VarianceThreshold(threshold=0.01)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _get_molecular_graph(self, mol) -> nx.Graph:
        """构建仅含共价键的分子图"""
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
        """从氨基酸序列提取特征"""
        return {
            "Cys_Pairs": sequence.count("C") // 2,
            "Proline_Kinks": sequence.count("P"),
            "Charge_Density": sum(1 for aa in sequence if aa in ["R", "K", "D", "E"]) / max(1, len(sequence)),
            "Hydrophobic_Ratio": sum(1 for aa in sequence if aa in ["A", "V", "L", "I", "M", "F", "W", "Y"]) / max(1,
                                                                                                                   len(sequence))
        }

    def _extract_molecular_features(self, mol) -> Dict:
        """从RDKit分子对象提取特征"""
        features = {}

        # 1. Morgan指纹
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
            self.logger.warning(f"Morgan指纹计算失败: {str(e)}")

        # 2. 分子描述符
        for desc_name, desc_func in self.feature_config["key_descriptors"].items():
            try:
                features[desc_name] = desc_func(mol)
            except Exception as e:
                features[desc_name] = 0
                self.logger.warning(f"计算描述符 {desc_name} 失败: {str(e)}")

        # 3. 官能团计数
        for group_name, smarts in self.feature_config["functional_groups"].items():
            try:
                pat = Chem.MolFromSmarts(smarts)
                features[f"fg_{group_name}"] = len(mol.GetSubstructMatches(pat)) if pat else 0
            except Exception as e:
                features[f"fg_{group_name}"] = 0
                self.logger.warning(f"官能团 {group_name} 匹配失败: {str(e)}")

        # 4. 分子图特征
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
            self.logger.warning(f"分子图特征计算失败: {str(e)}")

        return features

    def _get_feature_order(self):
        """生成特征顺序列表"""
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
        """处理单个FASTA文件，返回特征和标签"""
        sequences = []
        labels = []

        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # 精确的标签分配逻辑
                    if any(line.startswith(prefix) for prefix in self.positive_prefixes):
                        current_label = 1
                        self.logger.debug(f"识别为正样本: {line}")
                    elif any(line.startswith(prefix) for prefix in self.negative_prefixes):
                        current_label = 0
                        self.logger.debug(f"识别为负样本: {line}")
                    else:
                        # 如果既不符合正样本也不符合负样本格式，发出警告并使用默认标签
                        default_label = 1 if "pos" in output_filename.lower() else 0
                        self.logger.warning(f"无法识别的文件头格式: {line} | 使用默认标签: {default_label}")
                        current_label = default_label
                elif line:  # 非空行且不以>开头的是序列
                    sequences.append(line)
                    labels.append(current_label)

        # 验证标签分配
        unique_labels = set(labels)
        if len(unique_labels) != 2 and len(sequences) > 0:
            self.logger.error(f"警告: 文件 {fasta_path} 中只找到一种标签: {unique_labels}")

        features = []
        feature_order = self._get_feature_order()
        failed_count = 0

        for seq in sequences:
            try:
                mol = Chem.MolFromSequence(seq)
                if not mol:
                    raise ValueError("无效的肽序列")

                Chem.AddHs(mol)
                mol_feats = self._extract_molecular_features(mol)
                seq_feats = self._extract_sequence_features(seq)

                # 合并特征并保持固定顺序
                combined = {**mol_feats, **seq_feats}
                features.append([combined.get(k, 0) for k in feature_order])

            except Exception as e:
                failed_count += 1
                self.logger.error(f"处理序列失败: {seq[:10]}... | 错误: {str(e)}")
                features.append(np.zeros(len(feature_order)))

        feature_array = np.array(features, dtype=np.float32)
        label_array = np.array(labels, dtype=np.int32)

        # 保存结果
        feature_output_path = os.path.join(self.output_dir, output_filename)
        label_output_path = os.path.join(self.output_dir, output_filename.replace("features", "labels"))

        np.save(feature_output_path, feature_array)
        np.save(label_output_path, label_array)

        self.logger.info(
            f"处理完成 | 文件: {os.path.basename(fasta_path)} | 成功: {len(sequences) - failed_count} | "
            f"失败: {failed_count} | 特征维度: {feature_array.shape[1]} | "
            f"标签分布 - 正样本: {sum(labels)}, 负样本: {len(labels) - sum(labels)}"
        )

    def fit_transform(self, train_datasets: Dict[str, str]):
        """
        在训练数据上拟合特征选择器和标准化器

        Args:
            train_datasets: 训练数据集路径字典 {名称: 路径}
        """
        if self.is_fitted:
            return

        self.logger.info("开始在训练数据上拟合特征选择器和标准化器...")

        # 收集所有训练特征
        train_features = []
        for name, path in train_datasets.items():
            output_filename = f"{name}_features.npy"
            self._process_single_file(path, output_filename)

            # 加载刚保存的特征
            feature_path = os.path.join(self.output_dir, output_filename)
            train_features.append(np.load(feature_path))

        # 合并所有训练特征
        X_train = np.concatenate(train_features)

        # 拟合特征选择器和标准化器
        X_train_selected = self.selector.fit_transform(X_train)
        self.scaler.fit(X_train_selected)

        self.is_fitted = True
        self.logger.info(
            f"特征选择器和标准化器拟合完成 | 原始特征: {X_train.shape[1]} | 选择后特征: {X_train_selected.shape[1]}")

    def transform(self, dataset_name: str, dataset_path: str):
        """
        对数据集应用特征转换

        Args:
            dataset_name: 数据集名称
            dataset_path: 数据集路径
        """
        if not self.is_fitted:
            raise RuntimeError("特征提取器尚未拟合，请先调用fit_transform()方法")

        # 处理数据集
        output_filename = f"{dataset_name}_features.npy"
        self._process_single_file(dataset_path, output_filename)

        # 加载特征并应用转换
        feature_path = os.path.join(self.output_dir, output_filename)
        features = np.load(feature_path)

        # 应用特征选择和标准化
        features_selected = self.selector.transform(features)
        features_scaled = self.scaler.transform(features_selected)

        # 保存转换后的特征
        output_path = os.path.join(self.output_dir, f"{dataset_name}_features_transformed.npy")
        np.save(output_path, features_scaled)
        self.logger.info(f"已保存转换后的特征到: {output_path}")

    def save_extractor(self):
        """保存特征提取器配置（可pickle版本）"""
        save_path = os.path.join(self.output_dir, "topo_feature_extractor.pkl")

        # 创建可pickle的配置字典
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
        self.logger.info(f"特征提取器配置已保存到: {save_path}")


if __name__ == "__main__":
    # 配置路径
    DATA_DIR = "/home/kanglq/code_file/MyProject/Features_model/Data/fasta"
    OUTPUT_DIR = "/home/kanglq/code_file/MyProject/Features_model/Features_results/Topo_Feature"

    # 初始化提取器
    extractor = TopologicalFeatureExtractor(
        output_dir=OUTPUT_DIR,
        log_level="DEBUG"  # 使用DEBUG级别查看更多细节
    )

    # 定义数据集路径
    train_datasets = {
        "train_pos": os.path.join(DATA_DIR, "Train_Data/train_pos.fasta"),
        "train_neg": os.path.join(DATA_DIR, "Train_Data/train_neg.fasta")
    }

    test_datasets = {
        "test_pos": os.path.join(DATA_DIR, "Test_Data/test_pos.fasta"),
        "test_neg": os.path.join(DATA_DIR, "Test_Data/test_neg.fasta")
    }

    independent_datasets = {
        "independent_pos": os.path.join(DATA_DIR, "Independent_Data/independent_pos.fasta"),
        "independent_neg": os.path.join(DATA_DIR, "Independent_Data/independent_neg.fasta")
    }

    # 1. 在训练数据上拟合特征选择器和标准化器
    extractor.fit_transform(train_datasets)

    # 2. 转换训练数据
    for name, path in train_datasets.items():
        extractor.transform(name, path)

    # 3. 转换测试数据
    for name, path in test_datasets.items():
        extractor.transform(name, path)

    # 4. 转换独立数据集
    for name, path in independent_datasets.items():
        extractor.transform(name, path)

    # 保存提取器状态
    extractor.save_extractor()