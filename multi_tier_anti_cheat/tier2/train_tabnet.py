import os
import argparse
import time

import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import sys
from pathlib import Path

# Ensure this file's directory is on sys.path to allow importing data_utils when run directly
FILE_DIR = Path(__file__).resolve().parent
if str(FILE_DIR) not in sys.path:
    sys.path.append(str(FILE_DIR))

from data_utils import (
    prepare_data,
    ensure_dir,
    compute_metrics,
    tune_threshold,
    plot_and_save_curves,
    save_json,
)


def compute_class_weights(y: np.ndarray):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    weight_pos = float(neg / max(pos, 1))
    return [1.0, weight_pos]


def train(args):
    out_dir = os.path.join("outputs", "tabnet")
    ensure_dir(out_dir)

    splits = prepare_data()

    clf = TabNetClassifier(
        n_d=args.n_d,
        n_a=args.n_a,
        n_steps=args.n_steps,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        momentum=0.02,
        lambda_sparse=1e-4,
        seed=42,
        verbose=0,
        device_name="cpu" if args.cpu else "auto",
        optimizer_params=dict(lr=args.lr),
    )

    class_weights = compute_class_weights(splits.y_train)
    # Convert class weights to per-sample weights expected by TabNet
    sample_weights = np.where(splits.y_train == 1, class_weights[1], class_weights[0]).astype(np.float32)

    clf.fit(
        X_train=splits.X_train,
        y_train=splits.y_train,
        eval_set=[(splits.X_val, splits.y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        virtual_batch_size=min(128, args.batch_size),
        weights=sample_weights,
    )

    # Threshold tuning on val
    pv = clf.predict_proba(splits.X_val)[:, 1]
    best_thr = tune_threshold(splits.y_val, pv, metric="precision")

    # Test eval
    pt = clf.predict_proba(splits.X_test)[:, 1]
    metrics = compute_metrics(splits.y_test, pt, threshold=best_thr)

    # Save
    ts = int(time.time())
    clf.save_model(os.path.join(out_dir, f"tabnet_{ts}"))

    plot_and_save_curves(splits.y_test, pt, out_dir, prefix="tabnet")
    save_json(metrics, os.path.join(out_dir, f"metrics_{ts}.json"))

    print("Test metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TabNet for cheat detection")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--n_d", type=int, default=16)
    parser.add_argument("--n_a", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)
