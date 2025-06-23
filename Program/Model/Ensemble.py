"""
Ensemble model framework combining DNN and multiple tree-based models
for peptide classification tasks. Includes attention-enhanced DNN,
stacked generalization, and feature group weighting.
"""

import os
import torch
import joblib
import numpy as np
import pandas as pd
from collections import OrderedDict
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve

# ------------------------ Enhanced DNN ------------------------
class EnhancedDNN(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x)
        features = self.feature_extractor(x)
        weights = self.attention(features)
        output = self.output_layer(features * weights)
        return output.squeeze()

    def predict_proba(self, X, device='cpu'):
        self.eval()
        X_tensor = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            return self(X_tensor).cpu().numpy()

# ------------------------ Optimized Ensemble ------------------------
class AdvancedOptimizedEnsemble:
    def __init__(self, base_models, metric='mcc', n_folds=5):
        self.models = base_models
        self.metric = metric
        self.n_folds = n_folds
        self.weights = {}
        self.thresholds = {}
        self.meta_model = LGBMClassifier(n_estimators=200)

    def fit(self, X, y):
        for name, model in self.models.items():
            if name != 'DNN':
                model.fit(X, y)
            else:
                model.fit(X, y)
        self.weights = {name: 1 / len(self.models) for name in self.models}  # equal weighting
        self.meta_model.fit(self._stacked_features(X), y)

    def predict_proba(self, X):
        weighted = np.zeros(len(X))
        for name, model in self.models.items():
            proba = model.predict_proba(X) if name != 'DNN' else model.predict_proba(X)
            weighted += self.weights[name] * proba
        meta = self.meta_model.predict_proba(self._stacked_features(X))[:, 1]
        return 0.6 * meta + 0.4 * weighted

    def _stacked_features(self, X):
        return np.vstack([
            model.predict_proba(X) if name != 'DNN' else model.predict_proba(X)
            for name, model in self.models.items()
        ]).T

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

# ------------------------ Utility ------------------------
def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-6)

def evaluate(y_true, y_pred, y_prob):
    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'F1': f1_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Specificity': specificity_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

# ------------------------ Main (example usage) ------------------------
if __name__ == "__main__":
    # Example: initialize dummy models (replace with trained ones)
    input_dim = 100
    dummy_data = np.random.rand(200, input_dim)
    dummy_label = np.random.randint(0, 2, 200)

    base_models = {
        'DNN': EnhancedDNN(input_dim),
        'RandomForest': RandomForestClassifier(),
        'LightGBM': LGBMClassifier(),
        'XGBoost': XGBClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0),
        'GradientBoosting': GradientBoostingClassifier()
    }

    ensemble = AdvancedOptimizedEnsemble(base_models)
    ensemble.fit(dummy_data, dummy_label)

    pred = ensemble.predict(dummy_data)
    prob = ensemble.predict_proba(dummy_data)
    results = evaluate(dummy_label, pred, prob)
    print(results)
