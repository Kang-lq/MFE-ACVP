import os
import torch
import numpy as np
import pandas as pd
import pickle
import joblib
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
# 设置使用第 2 个 GPU（索引从 0 开始）
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 然后在您的设备选择代码中保持原样
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的设备: {device}")
class SecondaryStructureFeatureExtractor:
    def __init__(self):
        # 定义数据集路径（更新为新的路径结构）
        self.datasets = {
            'train_pos': '/home/kanglq/code_file/MyProject/Features_model/Data/CSV/train_pos',
            'train_neg': '/home/kanglq/code_file/MyProject/Features_model/Data/CSV/train_neg',
            'test_pos': '/home/kanglq/code_file/MyProject/Features_model/Data/CSV/test_pos',
            'test_neg': '/home/kanglq/code_file/MyProject/Features_model/Data/CSV/test_neg',
            'independent_pos': '/home/kanglq/code_file/MyProject/Features_model/Data/CSV/independent_pos',
            'independent_neg': '/home/kanglq/code_file/MyProject/Features_model/Data/CSV/independent_neg'
        }
        self.output_dir = '/home/kanglq/code_file/MyProject/Features_model/Features_results/Struc_Feature/Secondary_Feature'
        os.makedirs(self.output_dir, exist_ok=True)

        # 参数配置（训练阶段确定的参数）
        self.confidence_threshold = None  # 将在fit中确定
        self.short_peptide_length = 15
        self.fitted_ = False  # 标记是否已完成拟合

        # 固定参数（不需要拟合）
        self.kd_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }

        # 新增：位置特征编码器
        self.position_encoder = OneHotEncoder(
            categories=[['none', 'N-term', 'middle', 'C-term']],
            handle_unknown='ignore'
        )

    def _determine_threshold(self, train_dfs):
        """基于训练数据确定最佳置信度阈值"""
        all_confidences = []
        for df in train_dfs:
            conf = df['p[q3_H]'] + df['p[q3_E]'] + df['p[q3_C]']
            all_confidences.extend(conf.values)

        # 使用训练数据的中位数作为阈值
        self.confidence_threshold = np.median(all_confidences)
        print(f"✅ 确定置信度阈值为: {self.confidence_threshold:.3f} (基于训练数据)")

    def _adjust_threshold(self, seq_length):
        """动态调整置信度阈值（基于训练确定的阈值）"""
        if not self.fitted_:
            raise RuntimeError("请先调用fit()方法训练特征提取器！")

        if seq_length < self.short_peptide_length:
            return max(0.5, self.confidence_threshold - 0.02 * (self.short_peptide_length - seq_length))
        return self.confidence_threshold

    def fit(self):
        """仅在训练集上拟合参数"""
        if self.fitted_:
            print("⚠️ 特征提取器已拟合，跳过重复拟合")
            return

        print("=" * 50)
        print("🏋️ 开始训练阶段（仅使用train_pos和train_neg）")

        # 加载所有训练数据（严格只使用训练集）
        train_dfs = []
        for dataset in ['train_pos', 'train_neg']:
            input_dir = self.datasets[dataset]
            for filename in os.listdir(input_dir):
                if filename.endswith('.csv'):
                    df = self._load_csv(os.path.join(input_dir, filename))
                    if df is not None:
                        train_dfs.append(df)

        # 确定关键参数
        self._determine_threshold(train_dfs)

        # 新增：拟合位置特征编码器
        dummy_positions = [['N-term'], ['middle'], ['C-term'], ['none']]
        self.position_encoder.fit(dummy_positions)

        self.fitted_ = True

        # 保存特征提取器
        self._save_extractor()
        print("=" * 50 + "\n")

    def _load_csv(self, csv_path):
        """加载并验证CSV文件"""
        try:
            df = pd.read_csv(csv_path, sep=',', header=0)
            df.columns = df.columns.str.strip()

            required_columns = ['seq', 'q3', 'p[q3_H]', 'p[q3_E]', 'p[q3_C]']
            if not all(col in df.columns for col in required_columns):
                print(f"⚠️ 文件 {os.path.basename(csv_path)} 缺少必要列，已跳过")
                return None

            return df
        except Exception as e:
            print(f"⚠️ 加载文件 {os.path.basename(csv_path)} 出错: {str(e)}")
            return None

    def _calculate_global_stats(self, df):
        """计算全局统计特征"""
        threshold = self._adjust_threshold(len(df))
        valid_df = df[(df['p[q3_H]'] + df['p[q3_E]'] + df['p[q3_C]']) >= threshold]

        if len(valid_df) == 0:
            return {
                'H_percent': 0.0, 'E_percent': 0.0, 'C_percent': 0.0,
                'dominant_strength': 0.0
            }

        h_percent = (valid_df['q3'] == 'H').mean()
        e_percent = (valid_df['q3'] == 'E').mean()
        c_percent = (valid_df['q3'] == 'C').mean()

        dominant_strength = max(
            valid_df['p[q3_H]'].mean(),
            valid_df['p[q3_E]'].mean(),
            valid_df['p[q3_C]'].mean()
        )

        return {
            'H_percent': float(h_percent),
            'E_percent': float(e_percent),
            'C_percent': float(c_percent),
            'dominant_strength': float(dominant_strength)
        }

    def _calculate_fragment_features(self, df):
        """计算片段分布特征"""
        ss_seq = df['q3'].values
        max_h = current_h = max_e = current_e = 0
        h_starts = []
        e_starts = []

        for i, ss in enumerate(ss_seq):
            if ss == 'H':
                current_h += 1
                current_e = 0
                if current_h == 1:
                    h_starts.append(i)
                max_h = max(max_h, current_h)
            elif ss == 'E':
                current_e += 1
                current_h = 0
                if current_e == 1:
                    e_starts.append(i)
                max_e = max(max_e, current_e)
            else:
                current_h = current_e = 0

        def _position_feature(starts, length):
            if not starts:
                return {'pos': 'none', 'ratio': 0.0}
            pos = starts[0] / length
            if pos < 0.33:
                return {'pos': 'N-term', 'ratio': float(pos)}
            elif pos > 0.66:
                return {'pos': 'C-term', 'ratio': float(pos)}
            else:
                return {'pos': 'middle', 'ratio': float(pos)}

        h_pos = _position_feature(h_starts, len(df))
        e_pos = _position_feature(e_starts, len(df))

        def _calc_cv(ss_type):
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
            'max_H_length': float(max_h),
            'max_E_length': float(max_e),
            'H_position': h_pos['pos'],
            'E_position': e_pos['pos'],
            'H_position_ratio': float(h_pos['ratio']),
            'E_position_ratio': float(e_pos['ratio']),
            'H_length_CV': _calc_cv('H'),
            'E_length_CV': _calc_cv('E')
        }

    def _calculate_transition_matrix(self, df):
        """计算马尔可夫转移矩阵"""
        ss_types = ['H', 'E', 'C']
        trans_counts = np.zeros((3, 3))

        for i in range(len(df) - 1):
            current = df.iloc[i]['q3']
            next_ = df.iloc[i + 1]['q3']
            if current in ss_types and next_ in ss_types:
                trans_counts[ss_types.index(current), ss_types.index(next_)] += 1

        row_sums = trans_counts.sum(axis=1, keepdims=True)
        trans_matrix = np.divide(
            trans_counts,
            row_sums,
            out=np.zeros_like(trans_counts),
            where=row_sums != 0
        )

        try:
            eigenvalues, eigenvectors = np.linalg.eig(trans_matrix.T)
            stationary = eigenvectors[:, np.isclose(eigenvalues, 1)].real[:, 0]
            stationary /= stationary.sum()
        except:
            stationary = np.array([1 / 3, 1 / 3, 1 / 3])

        features = {}
        for i, from_ss in enumerate(ss_types):
            for j, to_ss in enumerate(ss_types):
                features[f'trans_{from_ss}_to_{to_ss}'] = float(trans_matrix[i, j])
            features[f'stat_{from_ss}'] = float(stationary[i])

        return features

    def _calculate_physicochemical(self, df):
        """计算物理化学特征"""
        helix_hydro = []
        current_seq = []
        for _, row in df.iterrows():
            if row['q3'] == 'H':
                current_seq.append(row['seq'])
            elif current_seq:
                hydro = np.mean([self.kd_scale.get(aa, 0) for aa in current_seq])
                helix_hydro.append(float(hydro))
                current_seq = []
        if current_seq:
            hydro = np.mean([self.kd_scale.get(aa, 0) for aa in current_seq])
            helix_hydro.append(float(hydro))

        polar_aas = {'D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y'}
        polar_pos = [i for i, row in df.iterrows()
                     if row['q3'] == 'E' and row['seq'] in polar_aas]
        if len(polar_pos) > 1:
            polar_clustering = float(1 / np.mean(np.diff(polar_pos)))
        else:
            polar_clustering = 0.0

        return {
            'avg_helix_hydro': float(np.mean(helix_hydro)) if helix_hydro else 0.0,
            'polar_clustering_sheet': float(polar_clustering)
        }

    def _encode_features(self, features):
        """编码特征为数值型数组"""
        # 提取位置特征并编码
        h_position = features.pop('H_position')
        e_position = features.pop('E_position')

        # 编码位置特征
        h_pos_encoded = self.position_encoder.transform([[h_position]]).toarray()[0]
        e_pos_encoded = self.position_encoder.transform([[e_position]]).toarray()[0]

        # 创建数值特征字典
        numeric_features = {}
        for k, v in features.items():
            if isinstance(v, (int, float, np.number)):
                numeric_features[k] = float(v)
            elif isinstance(v, str):
                numeric_features[k] = 0.0  # 处理意外字符串
            else:
                numeric_features[k] = float(v) if v is not None else 0.0

        # 添加编码后的位置特征
        for i, val in enumerate(h_pos_encoded):
            numeric_features[f'H_position_{i}'] = float(val)
        for i, val in enumerate(e_pos_encoded):
            numeric_features[f'E_position_{i}'] = float(val)

        return numeric_features

    def extract_features(self, csv_path):
        """从单个CSV文件提取特征并确保数值类型"""
        if not self.fitted_:
            raise RuntimeError("请先调用fit()方法训练特征提取器！")

        df = self._load_csv(csv_path)
        if df is None:
            return None

        # 计算原始特征
        raw_features = {}
        raw_features.update(self._calculate_global_stats(df))
        raw_features.update(self._calculate_fragment_features(df))  # 确保调用该方法
        raw_features.update(self._calculate_transition_matrix(df))
        raw_features.update(self._calculate_physicochemical(df))

        # 编码为数值型特征
        numeric_features = self._encode_features(raw_features)
        return numeric_features

    def process_dataset(self, dataset_name):
        """处理整个数据集并保存为数值数组"""
        if not self.fitted_ and dataset_name.startswith(('test', 'independent')):
            raise RuntimeError("请先调用fit()方法训练特征提取器！")

        input_dir = self.datasets[dataset_name]
        features = []
        labels = []

        for filename in tqdm(os.listdir(input_dir),
                             desc=f"📊 处理 {dataset_name}"):
            if filename.endswith('.csv'):
                csv_path = os.path.join(input_dir, filename)
                peptide_id = filename.split('.')[0]
                feat = self.extract_features(csv_path)
                if feat is not None:
                    features.append(list(feat.values()))
                    labels.append(1 if 'pos' in dataset_name else 0)

        feature_array = np.array(features, dtype=np.float32)
        label_array = np.array(labels, dtype=np.int32)

        # 保存数据
        feature_path = os.path.join(self.output_dir, f"{dataset_name}_features.npy")
        label_path = os.path.join(self.output_dir, f"{dataset_name}_labels.npy")
        np.save(feature_path, feature_array)
        np.save(label_path, label_array)

        print(f"💾 保存 {dataset_name} 特征到 {feature_path}")
        print(f"  - 特征矩阵形状: {feature_array.shape}")
        print(f"  - 标签矩阵形状: {label_array.shape}")

    def _save_extractor(self):
        """保存特征提取器"""
        extractor_path = os.path.join(self.output_dir, "feature_extractor.pkl")
        joblib.dump(self, extractor_path)
        print(f"💾 保存特征提取器到 {extractor_path}")

    @classmethod
    def load_extractor(cls, path):
        """加载特征提取器"""
        return joblib.load(path)

    def process_all_datasets(self):
        """完整的特征提取流程"""
        print("=" * 50)
        print("🚀 开始特征提取流程")

        # 1. 训练阶段（仅使用训练集）
        self.fit()

        # 2. 处理所有数据集（包括独立数据集）
        for dataset in self.datasets:
            self.process_dataset(dataset)

        print("=" * 50)
        print("🎉 特征提取完成！")


if __name__ == "__main__":
    # 初始化特征提取器
    extractor = SecondaryStructureFeatureExtractor()

    # 执行完整流程
    extractor.process_all_datasets()