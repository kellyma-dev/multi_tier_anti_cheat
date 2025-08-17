import os
import json
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
from pathlib import Path

current_dir = Path(__file__).resolve().parent
data_dir = os.path.join(current_dir, "../../data/RevStats/dataset/")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class DataSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: Optional[list] = None


def load_csv_splits(
    train_file: str = "final_train.csv",
    val_file: str = "final_val.csv",
    test_file: str = "final_test.csv",
    drop_id_first_col: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test CSVs. Remove first row (header) and first columd (ID)."""
    train_df = pd.read_csv(os.path.join(data_dir, train_file))
    val_df = pd.read_csv(os.path.join(data_dir, val_file))
    test_df = pd.read_csv(os.path.join(data_dir, test_file))

    if drop_id_first_col:
        train_df = train_df.iloc[:, 1:]
        val_df = val_df.iloc[:, 1:]
        test_df = test_df.iloc[:, 1:]

    return train_df, val_df, test_df


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and label. Label is the last column."""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def preprocess_numeric(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer, StandardScaler, list]:
    """Median impute and standardize numeric features. Fit on train only."""
    feature_names = list(X_train.columns)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    return X_train_scaled, X_val_scaled, X_test_scaled, imputer, scaler, feature_names


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    """Tune probability threshold on validation set to maximize the desired metric (default F1)."""
    if metric not in {"f1", "precision", "recall"}:
        metric = "f1"
    best_thr, best_val = 0.5, -1
    thresholds = np.linspace(0.05, 0.95, 19)
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        val = {"f1": f1, "precision": precision, "recall": recall}[metric]
        if val > best_val:
            best_val, best_thr = val, thr
    return float(best_thr)


def plot_and_save_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: str,
    prefix: str,
) -> None:
    ensure_dir(out_dir)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png"), bbox_inches="tight")
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png"), bbox_inches="tight")
    plt.close()


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def prepare_data() -> DataSplits:
    train_df, val_df, test_df = load_csv_splits()
    X_tr, y_tr = split_features_labels(train_df)
    X_va, y_va = split_features_labels(val_df)
    X_te, y_te = split_features_labels(test_df)

    X_tr_s, X_va_s, X_te_s, _, _, feat_names = preprocess_numeric(X_tr, X_va, X_te)

    return DataSplits(
        X_train=X_tr_s.astype(np.float32),
        y_train=y_tr.to_numpy().astype(int),
        X_val=X_va_s.astype(np.float32),
        y_val=y_va.to_numpy().astype(int),
        X_test=X_te_s.astype(np.float32),
        y_test=y_te.to_numpy().astype(int),
        feature_names=feat_names,
    )
