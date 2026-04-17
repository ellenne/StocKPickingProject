"""
src/models/neural_models.py
───────────────────────────
Implements:
  - DNNModel  : Feedforward deep neural network (3 hidden layers 20-10-5)
  - LSTMModel : LSTM recurrent neural network (1 hidden layer, 30 cells)

Both models are implemented in PyTorch to avoid TensorFlow / Keras version
fragility.  The architectures exactly match the paper's Appendix Table 1:

DNN
───
  Input → Dense(20, ReLU) → BN → Dense(10, ReLU) → BN → Dense(5, ReLU) → BN
  → Dense(2, Softmax)
  Optimizer: RMSprop lr=0.001
  Regularisation: L1=0.0001, BatchNorm
  Epochs: 100, Early stopping patience=10

LSTM
────
  Input (seq_len, n_features) → LSTM(30) → Dense(2, Softmax)
  Optimizer: RMSprop lr=0.001
  Epochs: 100, Early stopping patience=10

Note on LSTM input shape
────────────────────────
The paper feeds sequences of weekly observations to the LSTM.  Here we use
the last *seq_len* weeks of features as the sequence for each prediction.
When training we construct sequences from the cross-section of stocks at each
week.  At test time a single week's features are padded with the previous
``seq_len - 1`` weeks' cross-sectionally standardised features.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseStockModel

logger = logging.getLogger(__name__)


def _to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32))
    return torch.tensor(np.asarray(x, dtype=np.float32), dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  DNN Architecture
# ─────────────────────────────────────────────────────────────────────────────

class _DNNNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden: list[int],
        l1_reg: float,
        batch_norm: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.l1_reg = l1_reg
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def l1_loss(self) -> torch.Tensor:
        """L1 regularisation term over all weight matrices."""
        return self.l1_reg * sum(
            p.abs().sum() for p in self.parameters() if p.ndim >= 2
        )


class DNNModel(BaseStockModel):
    """Paper DNN: 3 hidden layers (20-10-5), BatchNorm, L1 reg, RMSprop."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        mc = cfg._raw["models"]["dnn"]
        self._hidden = mc.get("hidden_layers", [20, 10, 5])
        self._epochs = mc.get("epochs", 100)
        self._patience = mc.get("early_stopping_patience", 10)
        self._lr = mc.get("learning_rate", 0.001)
        self._l1 = mc.get("l1_reg", 0.0001)
        self._bn = mc.get("batch_norm", True)
        self._dropout = mc.get("dropout", 0.0)
        self._batch_size = mc.get("batch_size", 512)
        self._seed = mc.get("seed", cfg.seed)
        self._net: _DNNNet | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "dnn"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        n_features = np.asarray(X_train).shape[1]
        self._net = _DNNNet(
            n_features, self._hidden, self._l1, self._bn, self._dropout
        ).to(self._device)

        X_tr = _to_tensor(X_train)
        y_tr = _to_tensor(y_train, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=self._batch_size,
            shuffle=True,
        )

        if X_val is not None and y_val is not None:
            X_v = _to_tensor(X_val).to(self._device)
            y_v = _to_tensor(y_val, dtype=torch.long).to(self._device)
            has_val = True
        else:
            has_val = False

        opt = torch.optim.RMSprop(self._net.parameters(), lr=self._lr)
        loss_fn = nn.CrossEntropyLoss()
        best_val_loss = np.inf
        patience_cnt = 0
        best_state = None

        logger.info("[%s] training %d epochs …", self.name, self._epochs)
        for epoch in range(self._epochs):
            self._net.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                logits = self._net(Xb)
                loss = loss_fn(logits, yb) + self._net.l1_loss()
                loss.backward()
                opt.step()

            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_logits = self._net(X_v)
                    val_loss = loss_fn(val_logits, y_v).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= self._patience:
                        logger.info("[%s] early stopping at epoch %d", self.name, epoch + 1)
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        self._net.eval()
        X_t = _to_tensor(X).to(self._device)
        with torch.no_grad():
            logits = self._net(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


# ─────────────────────────────────────────────────────────────────────────────
#  LSTM Architecture
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_units: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_units,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_units, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # take the last time step
        return self.fc(self.dropout(last))


class LSTMModel(BaseStockModel):
    """Paper LSTM: 1 hidden layer with 30 cells, RMSprop lr=0.001."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        mc = cfg._raw["models"]["lstm"]
        self._hidden_units = mc.get("hidden_units", 30)
        self._seq_len = mc.get("seq_len", 8)
        self._epochs = mc.get("epochs", 100)
        self._patience = mc.get("early_stopping_patience", 10)
        self._lr = mc.get("learning_rate", 0.001)
        self._dropout = mc.get("dropout", 0.0)
        self._batch_size = mc.get("batch_size", 256)
        self._seed = mc.get("seed", cfg.seed)
        self._net: _LSTMNet | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "lstm"

    def _build_sequences(
        self, X: np.ndarray, seq_len: int
    ) -> np.ndarray:
        """Convert a 2-D feature matrix into (n_valid, seq_len, n_features) sequences."""
        n, f = X.shape
        if n < seq_len:
            # Pad with zeros at the front
            pad = np.zeros((seq_len - n, f), dtype=np.float32)
            X = np.vstack([pad, X])
            n = seq_len
        seqs = np.stack(
            [X[i : i + seq_len] for i in range(n - seq_len + 1)],
            axis=0,
        )
        return seqs.astype(np.float32)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        X_np = np.asarray(X_train, dtype=np.float32)
        y_np = np.asarray(y_train, dtype=np.int64)
        seqs = self._build_sequences(X_np, self._seq_len)

        # Align labels: drop first (seq_len - 1) rows
        y_aligned = y_np[self._seq_len - 1 :]
        assert len(seqs) == len(y_aligned), "Sequence/label length mismatch"

        n_features = X_np.shape[1]
        self._net = _LSTMNet(n_features, self._hidden_units, self._dropout).to(
            self._device
        )

        loader = DataLoader(
            TensorDataset(_to_tensor(seqs), _to_tensor(y_aligned, torch.long)),
            batch_size=self._batch_size,
            shuffle=False,  # respect time ordering
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_v_np = np.asarray(X_val, dtype=np.float32)
            seqs_v = self._build_sequences(X_v_np, self._seq_len)
            y_v_np = np.asarray(y_val, dtype=np.int64)[self._seq_len - 1 :]
            X_v = _to_tensor(seqs_v).to(self._device)
            y_v = _to_tensor(y_v_np, torch.long).to(self._device)

        opt = torch.optim.RMSprop(self._net.parameters(), lr=self._lr)
        loss_fn = nn.CrossEntropyLoss()
        best_val_loss = np.inf
        patience_cnt = 0
        best_state = None

        logger.info("[%s] training %d epochs …", self.name, self._epochs)
        for epoch in range(self._epochs):
            self._net.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                loss = loss_fn(self._net(Xb), yb)
                loss.backward()
                opt.step()

            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self._net(X_v), y_v).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= self._patience:
                        logger.info("[%s] early stopping at epoch %d", self.name, epoch + 1)
                        break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_np = np.asarray(X, dtype=np.float32)
        seqs = self._build_sequences(X_np, self._seq_len)
        self._net.eval()
        X_t = _to_tensor(seqs).to(self._device)
        with torch.no_grad():
            logits = self._net(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        # Align back to original n_rows by prepending NaN rows for the padded head
        n_missing = len(X_np) - len(probs)
        if n_missing > 0:
            pad = np.full((n_missing, 2), 0.5)
            probs = np.vstack([pad, probs])
        return probs
