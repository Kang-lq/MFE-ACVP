import os
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report, log_loss,
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# 固定随机种子
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(random_seed)

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 1. 增强的DNN模型 ====================
class EnhancedDNN(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.3):
        super().__init__()

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输入层
        x = self.input_layer(x)

        # 特征提取
        features = self.feature_extractor(x)

        # 注意力权重
        attention_weights = self.attention(features)

        # 加权特征
        weighted_features = features * attention_weights

        # 输出
        output = self.output_layer(weighted_features)

        return output.squeeze()

    def fit(self, X, y, epochs=150, batch_size=64, lr=1e-4, verbose=True):
        self.to(device)

        # 转换数据为Tensor
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 优化器和损失函数
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.BCELoss()

        # 训练循环
        best_loss = float('inf')
        patience = 10
        no_improve = 0

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)  # 梯度裁剪
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
                torch.save(self.state_dict(), 'best_dnn_model.pth')
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    self.load_state_dict(torch.load('best_dnn_model.pth'))
                    break

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    def predict_proba(self, X):
        with torch.no_grad():
            self.eval()
            if isinstance(X, pd.DataFrame):
                X = X.values  # Convert DataFrame to numpy array
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X).to(device)
            elif isinstance(X, torch.Tensor):
                X_tensor = X.to(device)
            else:
                raise ValueError("Input X must be a pandas DataFrame, numpy array or a torch tensor.")
            return self(X_tensor).cpu().numpy()


# ==================== 2. 完整的优化集成模型 ====================
class AdvancedOptimizedEnsemble:
    def __init__(self, models, metric='mcc', specificity_target=0.8, n_folds=5):
        self.models = models
        self.metric = metric
        self.specificity_target = specificity_target
        self.n_folds = n_folds
        self.weights = {}
        self.calibrators = {}
        self.feature_importances = None
        self.group_weights = None
        self.thresholds = {}
        self.meta_model = LGBMClassifier(n_estimators=200, max_depth=5, random_state=42)

    def calculate_feature_importance(self, X, y):
        """使用随机森林计算特征重要性"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        self.feature_importances = rf.feature_importances_
        return self.feature_importances

    def apply_feature_weighting(self, X, feature_groups):
        """应用特征分组加权"""
        if not isinstance(X, np.ndarray):
            X = X.values

        if self.group_weights is None:
            return X

        X_weighted = X.copy()
        for group, indices in feature_groups.items():
            # 确保索引不越界
            valid_indices = [i for i in indices if i < X.shape[1]]
            if valid_indices:
                X_weighted[:, valid_indices] = X_weighted[:, valid_indices] * self.group_weights[group]
        return X_weighted

    def calibrate_models(self, X, y, method='isotonic'):
        """校准模型输出概率"""
        self.calibrators = {}
        for name, model in self.models.items():
            if name == 'DNN':  # DNN已经有sigmoid输出
                self.calibrators[name] = model
            else:
                calibrator = CalibratedClassifierCV(model, method=method, cv=5)
                calibrator.fit(X, y)
                self.calibrators[name] = calibrator
        return self.calibrators

    def evaluate_model(self, model, X, y):
        """评估单个模型性能"""
        if isinstance(X, pd.DataFrame):
            X = X.values  # 将 DataFrame 转换为 numpy.ndarray

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)
            y_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        else:
            y_proba = model.predict(X)

        # 计算最优阈值
        precision, recall, thresholds = precision_recall_curve(y, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        # 计算各项指标
        y_pred = (y_proba > optimal_threshold).astype(int)

        metrics = {
            'auc': roc_auc_score(y, y_proba),
            'auprc': average_precision_score(y, y_proba),
            'f1': f1_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'specificity': specificity_score(y, y_pred),
            'mcc': matthews_corrcoef(y, y_pred),
            'threshold': optimal_threshold,
            'sen': sensitivity_score(y, y_pred)  # 添加SEN评估
        }
        return metrics

    def find_optimal_threshold(self, y_true, y_proba):
        """根据目标指标寻找最优阈值"""
        if self.metric == 'specificity':
            thresholds = np.linspace(0, 1, 100)
            best_thresh = 0.5
            for thresh in thresholds:
                y_pred = (y_proba > thresh).astype(int)
                spec = specificity_score(y_true, y_pred)
                if spec >= self.specificity_target:
                    best_thresh = thresh
                    break
            return best_thresh
        else:
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
            return thresholds[np.argmax(f1_scores)]

    def generate_meta_features(self, X, y=None):
        """生成Stacking元特征"""
        if y is None:
            # 测试模式
            X_meta = np.zeros((X.shape[0], len(self.calibrators)))
            for i, (name, model) in enumerate(self.calibrators.items()):
                if hasattr(model, "predict_proba"):
                    if isinstance(X, pd.DataFrame):
                        proba = model.predict_proba(X.values)
                    else:
                        proba = model.predict_proba(X)
                    # 修改这里：处理一维和二维概率输出
                    if proba.ndim == 2:
                        X_meta[:, i] = proba[:, 1]
                    else:
                        X_meta[:, i] = proba
                else:
                    X_meta[:, i] = model.predict(X.values if isinstance(X, pd.DataFrame) else X)
            return X_meta
        else:
            # 训练模式 - 使用交叉验证
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            X_meta = np.zeros((X.shape[0], len(self.calibrators)))

            for i, (name, model) in enumerate(self.calibrators.items()):
                print(f"Generating meta-features for {name}...")
                fold_preds = np.zeros(X.shape[0])

                for train_idx, val_idx in skf.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train = y[train_idx]

                    if name == 'DNN':
                        model_clone = EnhancedDNN(X_train.shape[1]).to(device)
                        model_clone.fit(X_train.values, y_train, epochs=50, verbose=False)
                        # 修改这里：DNN的predict_proba返回一维数组
                        proba = model_clone.predict_proba(X_val.values)
                        fold_preds[val_idx] = proba  # 直接赋值，不需要[:, 1]
                    else:
                        model_clone = clone(model)
                        if hasattr(model_clone, 'fit'):
                            model_clone.fit(X_train, y_train)

                        if hasattr(model_clone, "predict_proba"):
                            proba = model_clone.predict_proba(
                                X_val.values if isinstance(X_val, pd.DataFrame) else X_val)
                            # 处理一维和二维概率输出
                            if proba.ndim == 2:
                                fold_preds[val_idx] = proba[:, 1]
                            else:
                                fold_preds[val_idx] = proba
                        else:
                            fold_preds[val_idx] = model_clone.predict(
                                X_val.values if isinstance(X_val, pd.DataFrame) else X_val)

                X_meta[:, i] = fold_preds

            return X_meta

    def fit(self, X, y, feature_groups=None, calibration=True):
        """训练集成模型"""
        # 1. 特征重要性分析
        if feature_groups is not None:
            self.calculate_feature_importance(X, y)
            self.group_weights = {}
            for group, indices in feature_groups.items():
                # 确保索引不越界
                valid_indices = [i for i in indices if i < X.shape[1]]
                if valid_indices:
                    self.group_weights[group] = np.mean(self.feature_importances[valid_indices]) * 2 + 0.5
            X_weighted = self.apply_feature_weighting(X, feature_groups)
            X_weighted = pd.DataFrame(X_weighted, columns=X.columns)
        else:
            X_weighted = X

        # 2. 数据分割
        X_train, X_val, y_train, y_val = train_test_split(
            X_weighted, y, test_size=0.2, stratify=y, random_state=42)

        # 3. 模型校准
        if calibration:
            print("\nCalibrating models...")
            self.calibrate_models(X_train, y_train)
            models_to_use = self.calibrators
        else:
            models_to_use = self.models

        # 4. 训练基模型并评估
        print("\nTraining and evaluating base models...")
        model_performances = {}
        for name, model in models_to_use.items():
            if name != 'DNN':  # DNN已经在外部训练
                print(f"Training {name}...")
                model.fit(X_train, y_train)

            print(f"Evaluating {name}...")
            metrics = self.evaluate_model(model, X_val, y_val)
            model_performances[name] = metrics
            self.thresholds[name] = metrics['threshold']

        # 5. 计算模型权重
        print("\nCalculating model weights...")
        total_performance = sum(perf[self.metric] for perf in model_performances.values())
        self.weights = {
            name: (perf[self.metric] / total_performance) ** 2.0  # 非线性放大
            for name, perf in model_performances.items()
        }

        # 确保最小权重
        min_weight = 0.05  # 每个模型至少5%权重
        self.weights = {k: max(v, min_weight) for k, v in self.weights.items()}

        # 归一化
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        # 打印模型权重
        print("\nModel Weights:")
        for name, weight in self.weights.items():
            print(f"{name}: {weight:.4f}")

        # 6. 训练元模型
        print("\nTraining meta-model...")
        X_meta = self.generate_meta_features(X_weighted, y)
        self.meta_model.fit(X_meta, y)

        # 打印模型权重
        print("\nModel Weights and Performance:")
        perf_df = pd.DataFrame(model_performances).T
        perf_df['Weight'] = pd.Series(self.weights)
        print(perf_df.round(4))

        return self

    def predict_proba(self, X, feature_groups=None):
        """预测概率"""
        # 特征加权
        if feature_groups is not None and self.group_weights is not None:
            X_weighted = self.apply_feature_weighting(X, feature_groups)
            X_weighted = pd.DataFrame(X_weighted, columns=X.columns)
        else:
            X_weighted = X

        # 加权集成预测
        weighted_proba = np.zeros(len(X_weighted))
        for name, model in self.calibrators.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_weighted)
                proba = proba[:, 1] if proba.ndim == 2 else proba
            else:
                proba = model.predict(X_weighted)
            weighted_proba += self.weights[name] * proba

        # Stacking预测
        X_meta = self.generate_meta_features(X_weighted)
        stacking_proba = self.meta_model.predict_proba(X_meta)[:, 1]

        # 结合两种预测
        final_proba = 0.6 * stacking_proba + 0.4 * weighted_proba
        return final_proba

    def predict(self, X, y_true=None, feature_groups=None):
        """预测类别"""
        probas = self.predict_proba(X, feature_groups)

        if y_true is not None:
            threshold = self.find_optimal_threshold(y_true, probas)
        else:
            threshold = 0.5

        return (probas > threshold).astype(int), probas

    def evaluate(self, X, y, feature_groups=None):
        """评估模型性能"""
        y_pred, y_proba = self.predict(X, y, feature_groups)

        metrics = {
            'auc': roc_auc_score(y, y_proba),
            'auprc': average_precision_score(y, y_proba),
            'f1': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'specificity': specificity_score(y, y_pred),
            'accuracy': accuracy_score(y, y_pred),
            'mcc': matthews_corrcoef(y, y_pred),
            'sen': sensitivity_score(y, y_pred),  # 添加SEN评估
            'threshold': self.find_optimal_threshold(y, y_proba)
        }

        # 打印分类报告
        print("\nClassification Report:")
        print(classification_report(y, y_pred))

        # 绘制混淆矩阵
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()

        # 绘制ROC和PR曲线
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(y, y_proba)
        plt.plot(recall, precision, label=f'AP = {metrics["auprc"]:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return metrics


# ==================== 3. 可视化功能 ====================
def plot_roc_curves(models, X_sets, y_sets, set_names, save_path=None):
    """绘制多个模型在不同数据集上的ROC曲线"""
    plt.figure(figsize=(18, 6))

    for i, (X, y, name) in enumerate(zip(X_sets, y_sets, set_names)):
        plt.subplot(1, 3, i + 1)

        for model_name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X.values if isinstance(X, pd.DataFrame) else X)
                y_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            else:
                y_proba = model.predict(X.values if isinstance(X, pd.DataFrame) else X)

            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name} Set')
        plt.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_ensemble_roc_curves(base_models, ensemble, X_sets, y_sets, set_names, save_path=None):
    """绘制集成模型和基础模型在所有数据集上的ROC曲线"""
    plt.figure(figsize=(18, 6))

    for i, (X, y, name) in enumerate(zip(X_sets, y_sets, set_names)):
        plt.subplot(1, 3, i + 1)

        # 绘制基础模型ROC曲线
        for model_name, model in base_models.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X.values if isinstance(X, pd.DataFrame) else X)
                y_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            else:
                y_proba = model.predict(X.values if isinstance(X, pd.DataFrame) else X)

            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, linestyle='--',
                     label=f'{model_name} (AUC = {roc_auc:.3f})')

        # 绘制集成模型ROC曲线
        y_proba = ensemble.predict_proba(X)
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.5, color='black',
                 label=f'Ensemble (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name} Set')
        plt.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'ensemble_roc_curves.png'),
                    dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_type_analysis(feature_importances, feature_names, feature_groups, save_path=None):
    """绘制不同特征类型对模型的影响"""
    group_data = []
    for group, indices in feature_groups.items():
        valid_indices = [i for i in indices if i < len(feature_importances)]
        if valid_indices:
            group_importance = np.mean(feature_importances[valid_indices])
            group_std = np.std(feature_importances[valid_indices])
            group_data.append({
                'Feature Group': group,
                'Mean Importance': group_importance,
                'Std Importance': group_std,
                'Count': len(valid_indices)
            })

    group_df = pd.DataFrame(group_data)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(group_df['Feature Group'], group_df['Mean Importance'],
                   yerr=group_df['Std Importance'], capsize=5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')

    plt.xlabel('Feature Group')
    plt.ylabel('Mean Importance Score')
    plt.title('Feature Group Importance Analysis')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'feature_group_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return group_df


def plot_top_features(feature_importances, feature_names, top_n=20, save_path=None):
    """可视化前N个重要特征"""
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False).head(top_n)

    plt.figure(figsize=(12, 8))
    bars = plt.barh(feature_df['Feature'], feature_df['Importance'], color='skyblue')
    plt.gca().invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}',
                 va='center', ha='left')

    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature Name')
    plt.title(f'Top {top_n} Important Features')
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'top_features.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return feature_df


# ==================== 4. 辅助函数 ====================
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-6)


def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn + 1e-6)


def load_and_preprocess_data(feature_dir):
    """加载和预处理数据"""
    data = {
        'X_train': np.load(os.path.join(feature_dir, 'X_train_selected.npy')),
        'y_train': np.load(os.path.join(feature_dir, 'y_train.npy')).astype(int),
        'X_test': np.load(os.path.join(feature_dir, 'X_test_selected.npy')),
        'y_test': np.load(os.path.join(feature_dir, 'y_test.npy')).astype(int),
        'X_independent': np.load(os.path.join(feature_dir, 'X_independent_selected.npy')),
        'y_independent': np.load(os.path.join(feature_dir, 'y_independent.npy')).astype(int)
    }

    quantile_transformer = joblib.load(os.path.join(feature_dir, 'quantile_transformer.pkl'))
    scaler = joblib.load(os.path.join(feature_dir, 'scaler.pkl'))

    for key in ['X_train', 'X_test', 'X_independent']:
        data[key] = scaler.transform(quantile_transformer.transform(data[key]))

    feature_names = [f"feature_{i}" for i in range(data['X_train'].shape[1])]
    data['X_train'] = pd.DataFrame(data['X_train'], columns=feature_names)
    data['X_test'] = pd.DataFrame(data['X_test'], columns=feature_names)
    data['X_independent'] = pd.DataFrame(data['X_independent'], columns=feature_names)

    return data, feature_names


def load_base_models(model_dirs, input_dim, device):
    """加载预训练的基础模型"""
    models = OrderedDict()

    if 'DNN' in model_dirs:
        dnn_model = EnhancedDNN(input_dim).to(device)
        dnn_model.load_state_dict(torch.load(os.path.join(model_dirs['DNN'], "final_dnn_model.pth")))
        models['DNN'] = dnn_model

    if 'CatBoost' in model_dirs:
        catboost_model = CatBoostClassifier()
        catboost_model.load_model(os.path.join(model_dirs['CatBoost'], "final_catboost_model.cbm"))
        models['CatBoost'] = catboost_model

    for name, path in model_dirs.items():
        if name in ['RandomForest', 'XGBoost', 'LightGBM', 'GradientBoosting']:
            try:
                model = joblib.load(os.path.join(path, "final_model.pkl"))
                models[name] = model
            except Exception as e:
                print(f"Error loading {name} model: {e}")

    return models


# ==================== 5. 主执行流程 ====================
def main():
    # 配置路径
    model_dirs = {
        'RandomForest': "/home/kanglq/code_file/MyProject/Classification_model/Model_result/RandomForest",
        'XGBoost': "/home/kanglq/code_file/MyProject/Classification_model/Model_result/XGBoost",
        'LightGBM': "/home/kanglq/code_file/MyProject/Classification_model/Model_result/LightGBM",
        'CatBoost': "/home/kanglq/code_file/MyProject/Classification_model/Model_result/CatBoost",
        'GradientBoosting': "/home/kanglq/code_file/MyProject/Classification_model/Model_result/GradientBoosting",
        'DNN': "/home/kanglq/code_file/MyProject/Classification_model/Model_result/DNN"
    }

    feature_dir = "/home/kanglq/code_file/MyProject/Features_model/Feature Fusion/feature_selection_results"
    save_dir = "/home/kanglq/code_file/MyProject/Classification_model/Model_result/Ensemble"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载数据
    print("Loading data...")
    data, feature_names = load_and_preprocess_data(feature_dir)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    X_independent, y_independent = data['X_independent'], data['y_independent']

    # 2. 定义特征分组
    feature_groups = {
        'Sequence': range(0, 15),
        'Structure': range(15, 55),
        'Topology': range(55, 85),
        'Evolution': range(85, 100)
    }

    # 3. 加载基础模型
    print("\nLoading base models...")
    base_models = load_base_models(model_dirs, len(feature_names), device)

    # 4. 绘制ROC曲线
    print("\nPlotting ROC curves...")
    plot_roc_curves(
        base_models,
        [X_train, X_test, X_independent],
        [y_train, y_test, y_independent],
        ['Train', 'Test', 'Independent'],
        save_dir
    )

    # 5. 特征类型分析 - 修复后的代码
    print("\nAnalyzing feature groups...")
    rf_model = base_models['RandomForest']

    # 获取原始模型（如果是校准后的模型）
    if hasattr(rf_model, 'base_estimator'):  # 校准后的模型
        original_rf = rf_model.base_estimator
        print("Using base estimator from calibrated model for feature importance")
    else:
        original_rf = rf_model
        print("Using original model for feature importance")

    # 确保有特征重要性属性
    if not hasattr(original_rf, 'feature_importances_'):
        print("Original model has no feature_importances_, training temporary RandomForest...")
        temp_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        temp_rf.fit(X_train, y_train)
        feature_importances = temp_rf.feature_importances_
    else:
        feature_importances = original_rf.feature_importances_

    group_df = plot_feature_type_analysis(feature_importances, feature_names, feature_groups, save_dir)
    group_df.to_csv(os.path.join(save_dir, "feature_group_importance.csv"), index=False)

    # 6. 可视化前20个重要特征
    print("\nVisualizing top features...")
    top_features_df = plot_top_features(feature_importances, feature_names, top_n=20, save_path=save_dir)
    if top_features_df is not None:
        top_features_df.to_csv(os.path.join(save_dir, "top_features.csv"), index=False)

    # 7. 训练优化集成模型
    print("\nTraining optimized ensemble model...")
    ensemble = AdvancedOptimizedEnsemble(
        base_models,
        metric='mcc',
        specificity_target=0.85,
        n_folds=5
    )

    ensemble.fit(
        X_train,
        y_train,
        feature_groups=feature_groups,
        calibration=True
    )

    # 8. 保存独立数据集的预测概率
    print("\nSaving independent set predictions...")
    ind_proba = ensemble.predict_proba(X_independent)
    proba_df = pd.DataFrame({
        'True_Label': y_independent,
        'Predicted_Probability': ind_proba
    })
    proba_df.to_csv(os.path.join(save_dir, 'prob.csv'), index=False)
    print(f"Saved independent set predictions to {os.path.join(save_dir, 'prob.csv')}")

    # 9. 绘制集成模型和基础模型的ROC曲线
    print("\nPlotting ensemble ROC curves...")
    plot_ensemble_roc_curves(
        base_models,
        ensemble,
        [X_train, X_test, X_independent],
        [y_train, y_test, y_independent],
        ['Train', 'Test', 'Independent'],
        save_dir
    )

    # 10. 评估所有模型在所有数据集上的性能并保存为CSV
    print("\nEvaluating all models on all datasets...")

    # 定义要评估的数据集
    datasets = {
        'Train': (X_train, y_train),
        'Test': (X_test, y_test),
        'Independent': (X_independent, y_independent)
    }

    # 准备收集所有结果的DataFrame
    all_results = []

    # 评估每个模型在每个数据集上的表现
    for model_name, model in base_models.items():
        for dataset_name, (X, y) in datasets.items():
            print(f"Evaluating {model_name} on {dataset_name} set...")

            try:
                # 评估模型
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X.values if isinstance(X, pd.DataFrame) else X)
                    y_proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                else:
                    y_proba = model.predict(X.values if isinstance(X, pd.DataFrame) else X)

                # 计算最优阈值
                precision, recall, thresholds = precision_recall_curve(y, y_proba)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]
                y_pred = (y_proba > optimal_threshold).astype(int)

                # 计算混淆矩阵（添加异常处理）
                try:
                    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
                except Exception as e:
                    print(f"Error calculating confusion matrix for {model_name} on {dataset_name}: {str(e)}")
                    tn, fp, fn, tp = 0, 0, 0, 0  # 默认值

                # 计算各项指标
                metrics = {
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'AUC': roc_auc_score(y, y_proba),
                    'AUPRC': average_precision_score(y, y_proba),
                    'F1': f1_score(y, y_pred),
                    'Precision': precision_score(y, y_pred),
                    'Recall(SEN)': recall_score(y, y_pred),
                    'Specificity': specificity_score(y, y_pred),
                    'Accuracy': accuracy_score(y, y_pred),
                    'MCC': matthews_corrcoef(y, y_pred),
                    'Threshold': optimal_threshold,
                    'TP': tp,
                    'FP': fp,
                    'TN': tn,
                    'FN': fn
                }
                all_results.append(metrics)
            except Exception as e:
                print(f"Error evaluating {model_name} on {dataset_name}: {str(e)}")
                continue

    # 评估集成模型在所有数据集上的表现
    for dataset_name, (X, y) in datasets.items():
        print(f"Evaluating Ensemble on {dataset_name} set...")
        try:
            # 使用集成模型的evaluate方法
            y_pred, y_proba = ensemble.predict(X, y, feature_groups)

            # 计算混淆矩阵
            try:
                tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            except Exception as e:
                print(f"Error calculating confusion matrix for Ensemble on {dataset_name}: {str(e)}")
                tn, fp, fn, tp = 0, 0, 0, 0  # 默认值

            # 计算各项指标
            metrics = {
                'Model': 'Ensemble',
                'Dataset': dataset_name,
                'AUC': roc_auc_score(y, y_proba),
                'AUPRC': average_precision_score(y, y_proba),
                'F1': f1_score(y, y_pred),
                'Precision': precision_score(y, y_pred),
                'Recall(SEN)': recall_score(y, y_pred),
                'Specificity': specificity_score(y, y_pred),
                'Accuracy': accuracy_score(y, y_pred),
                'MCC': matthews_corrcoef(y, y_pred),
                'Threshold': ensemble.find_optimal_threshold(y, y_proba),
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn
            }
            all_results.append(metrics)
        except Exception as e:
            print(f"Error evaluating Ensemble on {dataset_name}: {str(e)}")
            continue

    # 转换为DataFrame
    results_df = pd.DataFrame(all_results)

    # 重新排序列顺序
    column_order = [
        'Model', 'Dataset', 'AUC', 'AUPRC', 'F1', 'Precision',
        'Recall(SEN)', 'Specificity', 'Accuracy', 'MCC', 'Threshold',
        'TP', 'FP', 'TN', 'FN'
    ]
    results_df = results_df[column_order]

    # 按照数据集分组保存为三个CSV文件
    for dataset_name in datasets.keys():
        dataset_results = results_df[results_df['Dataset'] == dataset_name]
        # 保存到文件
        csv_path = os.path.join(save_dir, f'model_performance_{dataset_name}.csv')
        dataset_results.to_csv(csv_path, index=False)
        print(f"Saved performance results for {dataset_name} set to {csv_path}")

    # 保存集成模型
    print("\nSaving ensemble model...")
    joblib.dump(ensemble, os.path.join(save_dir, "optimized_ensemble_model.pkl"))

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()