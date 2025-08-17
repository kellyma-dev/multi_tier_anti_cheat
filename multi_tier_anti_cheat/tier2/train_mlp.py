import os
import argparse
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import sys
from pathlib import Path

# Ensure this file's directory is on sys.path to allow importing data_utils when run directly
FILE_DIR = Path(__file__).resolve().parent
if str(FILE_DIR) not in sys.path:
    sys.path.append(str(FILE_DIR))

from data_utils import (
    prepare_data,
    DataSplits,
    ensure_dir,
    compute_metrics,
    tune_threshold,
    plot_and_save_curves,
    save_json,
)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    # weight for class 1 = N_neg / N_pos, class 0 = 1.0
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    weight_pos = (neg / max(pos, 1)).item() if isinstance(neg, np.ndarray) else neg / max(pos, 1)
    return torch.tensor([1.0, float(weight_pos)], dtype=torch.float32)


def train(args):
    out_dir = os.path.join("outputs", "mlp")
    ensure_dir(out_dir)

    splits: DataSplits = prepare_data()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    X_tr = torch.tensor(splits.X_train, dtype=torch.float32)
    y_tr = torch.tensor(splits.y_train, dtype=torch.long)
    X_va = torch.tensor(splits.X_val, dtype=torch.float32)
    y_va = torch.tensor(splits.y_val, dtype=torch.long)
    X_te = torch.tensor(splits.X_test, dtype=torch.float32)
    y_te = torch.tensor(splits.y_test, dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_va, y_va)
    test_ds = TensorDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MLP(in_dim=X_tr.shape[1], hidden=tuple(args.hidden), dropout=args.dropout).to(device)

    class_weights = compute_class_weights(splits.y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = -1
    best_state = None
    patience = args.patience
    no_improve = 0

    def infer_proba(loader):
        model.eval()
        probs = []
        ys = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy()
                probs.append(prob)
                ys.append(yb.numpy())
        return np.concatenate(ys), np.concatenate(probs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            logits2 = torch.stack([torch.zeros_like(logits), logits], dim=1)  # [B,2]
            loss = criterion(logits2, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # validation AUC
        yv, pv = infer_proba(val_loader)
        val_auc = float(
            0.5 if len(np.unique(yv)) < 2 else __import__("sklearn.metrics").metrics.roc_auc_score(yv, pv)
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {"model": model.state_dict(), "epoch": epoch}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # Tune threshold on validation
    yv, pv = infer_proba(val_loader)
    best_thr = tune_threshold(yv, pv, metric="f1")

    # Evaluate on test
    yt, pt = infer_proba(test_loader)
    metrics = compute_metrics(yt, pt, threshold=best_thr)

    # Save artifacts
    ts = int(time.time())
    ckpt_path = os.path.join(out_dir, f"best_mlp_{ts}.pt")
    torch.save({"state_dict": model.state_dict(), "feature_names": splits.feature_names}, ckpt_path)

    plot_and_save_curves(yt, pt, out_dir, prefix="mlp")
    save_json(metrics, os.path.join(out_dir, f"metrics_{ts}.json"))

    print("Test metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP for cheat detection")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")
    args = parser.parse_args()
    train(args)
