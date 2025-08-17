# Tier 2 Models: MLP, TabNet, Wide & Deep

This folder provides training scripts to predict if a game player is a cheater using deep neural networks on tabular data.

Assumptions per `t2_model_prompt.md`:
- First row is header.
- First column is an ID and is dropped from training/validation/testing features.
- Last column is the binary label (0/1), where 1 indicates the positive class (assumed cheater).
- Features are treated as numeric. Missing values are median-imputed and standardized (fit on train only).

Data location:
- `data/RevStats/dataset/final_train.csv`
- `data/RevStats/dataset/final_val.csv`
- `data/RevStats/dataset/final_test.csv`

Outputs are written to `mtac-ml/tier2/outputs/<model_name>/` including:
- Best checkpoint
- `metrics_*.json` with ROC-AUC, PR-AUC, F1, precision, recall, confusion matrix
- ROC and PR curve plots

## Environment

Dependencies are declared in the project's `pyproject.toml`:
- torch, pytorch-tabnet, numpy, pandas, scikit-learn, matplotlib, tqdm

Install via Poetry from project root:
```bash
poetry install
```

## Common CLI Options
- `--data_dir`: Path to dataset directory (default `data/RevStats/dataset`)
- `--epochs`: Max epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--patience`: Early stopping patience (on validation AUC)
- `--cpu`: Force CPU

Threshold is tuned on the validation set to maximize F1, then metrics are reported on the test set.

## Train MLP
```bash
python mtac-ml/mtac-ml/tier2/train_mlp.py --data_dir data/RevStats/dataset
```
MLP options:
- `--hidden 128 64` (default)
- `--dropout 0.2`

## Train TabNet
```bash
python mtac-ml/mtac-ml/tier2/train_tabnet.py --data_dir data/RevStats/dataset
```
TabNet options:
- `--n_d 16 --n_a 16 --n_steps 4` (defaults)

## Train Wide & Deep
```bash
python mtac-ml/mtac-ml/tier2/train_widedeep.py --data_dir data/RevStats/dataset
```
Wide & Deep options:
- `--hidden 128 64` (deep tower sizes)
- `--dropout 0.2`

## Notes
- Class imbalance is handled by class-weighted loss.
- No categorical encodings are applied. If there are categorical columns, we can extend preprocessing with encoders/embeddings.
- We avoid copying code from public repos; TabNet is used via its library interface.
