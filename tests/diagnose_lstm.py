"""
Quick diagnostic: compare shuffle=False (broken) vs shuffle=True (fixed) LSTM
training curves on the actual features_panel data.

Usage: python tests/diagnose_lstm.py
"""
import sys, logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.features.preprocessing import FeaturePreprocessor, build_feature_matrix
from src.models.neural_models import _build_panel_sequences, _LSTMNet, _to_tensor

cfg = load_config("configs/fast.yaml")

# ── Load ONE window from actual data ─────────────────────────────────────────
print("Loading features_panel …")
panel = pd.read_parquet("data/processed/features_panel.parquet")
feat_cfg = cfg._raw["features"]
feature_panel = build_feature_matrix(
    panel,
    ffill_limit=feat_cfg["ffill_limit"],
    include_technical=feat_cfg["technical"],
    include_fundamental=feat_cfg["fundamental"],
    include_sector_dummies=feat_cfg["sector_dummies"],
)
target = panel["target"]

dates = feature_panel.index.get_level_values("date").unique().sort_values()
# Window 1: train 2015-2017, val last 20%
train_end   = pd.Timestamp("2017-12-27")
val_start   = pd.Timestamp("2017-07-05")  # ~last 20% of 3yr window
test_start  = pd.Timestamp("2018-01-03")
test_end    = pd.Timestamp("2018-12-26")

d = feature_panel.index.get_level_values("date")
train_mask = d <= train_end
val_mask   = (d >= val_start) & (d <= train_end)

X_tr_raw = feature_panel.loc[train_mask]
y_tr_raw = target.loc[train_mask].dropna()
X_tr_raw = X_tr_raw.loc[y_tr_raw.index]
X_v_raw  = feature_panel.loc[val_mask]
y_v_raw  = target.loc[val_mask].dropna()
X_v_raw  = X_v_raw.loc[y_v_raw.index]

prep = FeaturePreprocessor(
    winsorize=feat_cfg["winsorize"],
    winsorize_pct=feat_cfg["winsorize_pct"],
)
prep.fit(X_tr_raw)
X_tr = prep.transform(X_tr_raw)
X_v  = prep.transform(X_v_raw)
y_tr = y_tr_raw.values.astype(np.int64)
y_v  = y_v_raw.values.astype(np.int64)

SEQ_LEN   = cfg._raw["models"]["lstm"]["seq_len"]
N_FEAT    = X_tr.shape[1]
PATIENCE  = 15
LR        = 0.001
WD        = 1e-4
DROPOUT   = 0.2
EPOCHS    = 20

print(f"Train rows: {len(X_tr)}, Val rows: {len(X_v)}, Features: {N_FEAT}")

# ── Build sequences ───────────────────────────────────────────────────────────
y_ser = pd.Series(y_tr, index=X_tr.index)
seqs_all, labels_all, pos_all = _build_panel_sequences(X_tr, y_ser, SEQ_LEN)

val_index_set = set(map(tuple, X_v.index.tolist()))
train_index   = X_tr.index
is_val = np.array([tuple(train_index[p]) in val_index_set for p in pos_all])

seqs_t,  labels_t  = seqs_all[~is_val], labels_all[~is_val]
seqs_v,  labels_v  = seqs_all[ is_val], labels_all[ is_val]
print(f"Train seqs: {len(seqs_t)}, Val seqs: {len(seqs_v)}")

X_v_t = _to_tensor(seqs_v).to("cpu")
y_v_t = _to_tensor(labels_v, torch.long).to("cpu")
loss_fn = nn.CrossEntropyLoss()


def run_training(shuffle: bool, weight_decay: float, dropout: float,
                 l1_reg: float, lr: float, label: str):
    torch.manual_seed(42)
    net = _LSTMNet(N_FEAT, 30, dropout, l1_reg=l1_reg).to("cpu")
    loader = DataLoader(
        TensorDataset(_to_tensor(seqs_t), _to_tensor(labels_t, torch.long)),
        batch_size=256, shuffle=shuffle, drop_last=False,
    )
    opt = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = np.inf
    improvements = 0
    patience_cnt = 0

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  shuffle={shuffle}  wd={weight_decay}  l1={l1_reg}  dropout={dropout}  lr={lr}")
    print(f"{'='*65}")
    print(f"  {'Epoch':>5}  {'Train':>8}  {'Val':>8}  {'Gap':>8}  {'Best?'}")

    for epoch in range(1, EPOCHS + 1):
        net.train()
        tr_loss, nb = 0.0, 0
        for Xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(net(Xb), yb) + net.l1_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item(); nb += 1
        tr_loss /= max(nb, 1)

        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(X_v_t), y_v_t).item()

        is_best = vl < best_val
        if is_best:
            best_val = vl
            improvements += 1
            patience_cnt = 0
        else:
            patience_cnt += 1

        marker = " *" if is_best else "  "
        print(f"  {epoch:>5}  {tr_loss:>8.4f}  {vl:>8.4f}  {vl-tr_loss:>+8.4f}{marker}")

        if patience_cnt >= PATIENCE:
            print(f"  >> Early stop at epoch {epoch}, best_val={best_val:.4f}, improvements={improvements}")
            break

    print(f"  >> Final: best_val={best_val:.4f}, improvements={improvements}/{epoch}")
    return improvements


# ── Compare ───────────────────────────────────────────────────────────────────
impr_broken = run_training(
    shuffle=False, weight_decay=0.0, dropout=0.0, l1_reg=0.0, lr=0.001,
    label="BROKEN  (shuffle=False, no reg, no dropout)"
)

impr_fixed = run_training(
    shuffle=True, weight_decay=WD, dropout=DROPOUT, l1_reg=1e-4, lr=LR,
    label="FIXED   (shuffle=True, wd=1e-4, l1=1e-4, dropout=0.2)"
)

print(f"\n{'='*65}")
print(f"  Improvements:  broken={impr_broken}  fixed={impr_fixed}")
assert impr_fixed > impr_broken, \
    f"Fix didn't help! broken={impr_broken} fixed={impr_fixed}"
print("  DIAGNOSIS CONFIRMED: fix improves LSTM training convergence.")
