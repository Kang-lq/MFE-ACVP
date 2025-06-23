import os
import numpy as np
from Bio import SeqIO
from Bio.Align import substitution_matrices
import pickle
import joblib
import torch
import subprocess
from tempfile import NamedTemporaryFile
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from multiprocessing import Pool
import logging
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
# 设置使用第 2 个 GPU（索引从 0 开始）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 然后在您的设备选择代码中保持原样
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的设备: {device}")
# 配置日志
logging.basicConfig(
    filename='feature_extraction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class FeatureExtractor:
    def __init__(self, corona_db_path,
                 output_dir="/home/kanglq/code_file/MyProject/Features_model/Features_results/Evolu_Feature",
                 max_len=100, n_jobs=4, use_alphafold=False):
        self.corona_db_path = os.path.abspath(corona_db_path)
        self.output_dir = os.path.abspath(output_dir)
        self.max_len = max_len
        self.n_jobs = min(n_jobs, 8)
        self.use_alphafold = use_alphafold

        os.makedirs(self.output_dir, exist_ok=True)

        self._verify_input_file()

        self._diamond_executable = self._find_diamond()
        if self._diamond_executable is None:
            raise RuntimeError("DIAMOND未安装或不在PATH中")

        self._load_matrices()
        self._init_rbd_positions()
        self._prepare_diamond_db()

        self.adapter = HTTPAdapter(max_retries=Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        ))

        self.af_cache = {}

    def _verify_input_file(self):
        if not os.path.exists(self.corona_db_path):
            raise FileNotFoundError(f"数据库文件不存在: {self.corona_db_path}")

        if os.path.getsize(self.corona_db_path) == 0:
            raise ValueError("数据库文件为空！")

        with open(self.corona_db_path) as f:
            if not f.readline().startswith(">"):
                raise ValueError("不是有效的FASTA文件！")

    def _find_diamond(self):
        try:
            conda_path = os.path.expanduser("~/anaconda3/envs/py38_rdkit/bin/diamond")
            if os.path.exists(conda_path):
                return conda_path

            paths_to_try = [
                "/usr/local/bin/diamond",
                "/usr/bin/diamond",
                "diamond"
            ]

            for path in paths_to_try:
                try:
                    subprocess.run([path, "version"],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   check=True)
                    return os.path.abspath(path)
                except:
                    continue

            raise RuntimeError("未找到有效的DIAMOND可执行文件")

        except Exception as e:
            print(f"查找DIAMOND时出错: {str(e)}")
            raise

    def _load_matrices(self):
        self.std_aa = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.std_aa)}
        self.blosum62 = substitution_matrices.load("BLOSUM62")

    def _init_rbd_positions(self):
        self.rbd_positions = {
            i: 1.5
            for i in range(319, 541)
        }

    def _prepare_diamond_db(self):
        db_name = self.corona_db_path + ".dmnd"

        if os.path.exists(db_name):
            print(f"✅ 使用现有数据库: {db_name}")
            return

        print(f"🛠️ 构建DIAMOND数据库: {db_name}")
        try:
            result = subprocess.run(
                [self._diamond_executable, "makedb",
                 "--in", self.corona_db_path,
                 "-d", self.corona_db_path,
                 "--threads", str(max(1, self.n_jobs // 2)),
                 "--verbose"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            print("数据库构建日志:")
            print(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"❌ 数据库构建失败。错误详情：\n{e.stderr}")
            raise

    def _blosum_encode(self, sequence):
        encoded = np.zeros((self.max_len, 20))
        seq_str = str(sequence.seq)
        for i, aa in enumerate(seq_str[:self.max_len]):
            if aa in self.aa_to_idx:
                encoded[i] = [self.blosum62.get((aa, std_aa), 0)
                              for std_aa in self.std_aa]
                if i in self.rbd_positions:
                    encoded[i] *= self.rbd_positions[i]
        return encoded

    def _run_diamond_search(self, query_seq):
        try:
            with NamedTemporaryFile(mode='w+') as tmp_query, \
                    NamedTemporaryFile(mode='w+') as tmp_out:

                tmp_query.write(f">query\n{str(query_seq)[:1000]}")
                tmp_query.flush()

                cmd = [
                    self._diamond_executable,
                    "blastp",
                    "--query", tmp_query.name,
                    "--db", self.corona_db_path,
                    "--outfmt", "6",
                    "--out", tmp_out.name,
                    "--max-target-seqs", "10",
                    "--evalue", "1e-3",
                    "--id", "20",
                    "--threads", "1",
                    "--block-size", "2",
                    "--fast",
                    "--tmpdir", "/dev/shm"
                ]

                try:
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
                        timeout=300
                    )

                    tmp_out.seek(0)
                    hits = []
                    for line in tmp_out:
                        parts = line.strip().split('\t')
                        if len(parts) >= 5:
                            try:
                                hits.append((int(parts[3]), int(parts[4]), float(parts[2])))
                            except (ValueError, IndexError):
                                continue

                    if hits:
                        pssm = np.zeros((self.max_len, 20))
                        for start, end, ident in hits:
                            weight = ident / 100.0
                            for pos in range(start - 1, min(end, self.max_len)):
                                pssm[pos] += weight
                        return pssm / len(hits)

                except subprocess.TimeoutExpired:
                    logging.warning("DIAMOND比对超时（300秒），跳过当前序列")
                    return None

        except subprocess.CalledProcessError as e:
            logging.error(f"DIAMOND运行错误: {e.stderr}")
        except Exception as e:
            logging.error(f"DIAMOND运行时错误: {str(e)}")

        return None

    def _guess_uniprot_id(self, sequence):
        seq_str = str(sequence)
        if "MFVFLVLLPLVSSQCV" in seq_str:
            return "P0DTC2"
        return None

    def _get_conservation(self, sequence):
        hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y'}
        conserved = 0
        for aa in sequence[:self.max_len]:
            if aa in hydrophobic:
                conserved += 0.5
            elif aa in {'C', 'G', 'P'}:
                conserved += 0.3
        return conserved / self.max_len

    def _get_alphafold_confidence(self, sequence):
        if not self.use_alphafold:
            return None

        uniprot_id = self._guess_uniprot_id(sequence)
        if uniprot_id is None:
            return None

        if uniprot_id in self.af_cache:
            return self.af_cache[uniprot_id]

        try:
            url = f"https://alphafold.ebi.ac.uk/api/prediction/{quote(uniprot_id)}"
            with requests.Session() as session:
                session.mount("https://", self.adapter)
                response = session.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    conf = np.mean(data.get("plddt", [0]))
                    self.af_cache[uniprot_id] = conf
                    return conf
        except Exception as e:
            logging.warning(f"AlphaFold请求失败: {str(e)}")

        return None

    def _extract_features(self, seq_record):
        try:
            seq_str = str(seq_record.seq)

            blosum = self._blosum_encode(seq_record)
            conservation = self._get_conservation(seq_str)

            diamond_pssm = self._run_diamond_search(seq_str)
            af_conf = self._get_alphafold_confidence(seq_str) if self.use_alphafold else None

            features = []

            features.extend(blosum.flatten())

            if diamond_pssm is not None:
                features.extend(diamond_pssm.flatten() * 0.7)
            else:
                features.extend(np.zeros(blosum.size) * 0.5)

            if af_conf is not None:
                features.extend([af_conf] * 10)
            else:
                features.extend([conservation] * 10)

            features = np.array(features)
            if len(features) < self.max_len:
                features = np.pad(features, (0, self.max_len - len(features)), mode='constant')
            elif len(features) > self.max_len:
                features = features[:self.max_len]

            if not np.issubdtype(features.dtype, np.number):
                raise ValueError("特征数组包含非数值类型")

            logging.info(f"序列 {seq_record.id} 特征提取成功，特征长度: {len(features)}")
            return features

        except Exception as e:
            logging.error(f"序列 {seq_record.id} 特征提取失败: {str(e)}")
            return None

    def check_dependencies(self):
        print("=== 依赖项检查 ===")

        try:
            result = subprocess.run(
                [self._diamond_executable, "version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            print(f"✅ DIAMOND可用 (版本: {result.stdout.strip()})")
        except Exception as e:
            print(f"❌ DIAMOND检查失败: {str(e)}")
            return False

        try:
            requests.get("https://www.ebi.ac.uk", timeout=5)
            print("✅ 网络连接正常")
        except Exception as e:
            print(f"❌ 网络连接检查失败: {str(e)}")
            if self.use_alphafold:
                return False

        if not os.path.exists(self.corona_db_path + ".dmnd"):
            print("⚠️ DIAMOND数据库未构建，将尝试自动构建...")
            try:
                self._prepare_diamond_db()
                print("✅ 数据库构建成功")
            except Exception as e:
                print(f"❌ 数据库构建失败: {str(e)}")
                return False
        else:
            print("✅ DIAMOND数据库已存在")

        return True

    def save_features(self, dataset_name, sequences, labels):
        features = self._transform_sequences(sequences)
        output_path = os.path.join(self.output_dir, f"{dataset_name}_features.npy")
        np.save(output_path, features)
        label_path = os.path.join(self.output_dir, f"{dataset_name}_labels.npy")
        np.save(label_path, labels)
        print(f"✅ 已保存 {dataset_name} ({len(sequences)}样本) 到 {output_path}")

    def _transform_sequences(self, sequences):
        features = np.array(self._parallel_extract(self._extract_features, sequences))
        return features

    def _parallel_extract(self, func, sequences):
        with Pool(min(self.n_jobs, 4)) as pool:
            return list(tqdm(pool.imap(func, sequences), total=len(sequences)))

    def fit(self, train_sequences):
        if not self.check_dependencies():
            raise RuntimeError("依赖项检查失败，无法继续")

        print("🔄 特征提取中...")

        train_features = self._transform_sequences(train_sequences)
        self.scaler = StandardScaler().fit(train_features)

        joblib.dump(self, os.path.join(self.output_dir, "feature_extractor.pkl"))
        print("✅ 特征提取器训练完成")

    def transform(self, sequences):
        features = self._transform_sequences(sequences)
        return self.scaler.transform(features)


def load_fasta(path):
    return list(SeqIO.parse(path, "fasta"))


if __name__ == "__main__":
    # 定义数据集路径
    DATA_DIR = "/home/kanglq/code_file/MyProject/Features_model/Data/fasta"
    DB_PATH = "/home/kanglq/code_file/MyProject/Features_model/Data/coronavirus_db_db/coronavirus_db.fasta"

    # 加载数据集
    train_pos = load_fasta(os.path.join(DATA_DIR, "Train_Data/train_pos.fasta"))
    train_neg = load_fasta(os.path.join(DATA_DIR, "Train_Data/train_neg.fasta"))
    test_pos = load_fasta(os.path.join(DATA_DIR, "Test_Data/test_pos.fasta"))
    test_neg = load_fasta(os.path.join(DATA_DIR, "Test_Data/test_neg.fasta"))
    independent_pos = load_fasta(os.path.join(DATA_DIR, "Independent_Data/independent_pos.fasta"))
    independent_neg = load_fasta(os.path.join(DATA_DIR, "Independent_Data/independent_neg.fasta"))

    # 初始化特征提取器
    extractor = FeatureExtractor(
        corona_db_path=DB_PATH,
        n_jobs=4,
        use_alphafold=False
    )

    print("🔧 训练特征提取器...")
    extractor.fit(train_pos + train_neg)

    print("💾 保存特征文件...")
    extractor.save_features("train_pos", train_pos, np.ones(len(train_pos)))
    extractor.save_features("train_neg", train_neg, np.zeros(len(train_neg)))
    extractor.save_features("test_pos", test_pos, np.ones(len(test_pos)))
    extractor.save_features("test_neg", test_neg, np.zeros(len(test_neg)))
    extractor.save_features("independent_pos", independent_pos, np.ones(len(independent_pos)))
    extractor.save_features("independent_neg", independent_neg, np.zeros(len(independent_neg)))