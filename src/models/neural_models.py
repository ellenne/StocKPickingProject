"""
src/models/neural_models.py
───────────────────────────
Implements:
  - DNNModel  : Feedforward deep neural network (3 hidden layers 20-10-5)
  - LSTMModel : LSTM recurrent neural network (1 hidden layer, 30 cells)

Both models are implemented in PyTorch.  Architectures exactly match the paper's
Appendix Table 1 and Section 3.5.

DNN (Wolff & Echterling 2022, Section 3.5)
───────────────────────────────────────────
  Input → Linear(n, 20) → BN(20) → ReLU
        → Linear(20, 10) → BN(10) → ReLU
        → Linear(10, 5)  → BN(5)  → ReLU
        → Linear(5, 2)
  Logits passed to CrossEntropyLoss (numerically stable; equivalent to the
  Softmax the paper mentions but avoids log(softmax) double-computation).
  Optimizer : RMSprop lr=0.001
  Regularisation: L1=0.0001 applied to all weight matrices
  Epochs : 100, Early stopping patience=10 on validation loss
  Best weights restored at end of training

Training visibility
────────────────────
  fit() emits INFO-level progress every `log_every_n_epochs` epochs:
      [dnn] W1/7 ep 10/100  train=0.6921  val=0.6918  (best)
  After training:
      [dnn] W1/7 done – ep 47  best_val=0.6882  improved 3 times
  Full per-epoch history stored in `self.training_history_`
  (list of dicts: epoch, train_loss, val_loss, is_best).

LSTM
────
  Input (seq_len, n_features) → LSTM(30) → Dense(2)
  Optimizer: RMSprop lr=0.001
  Epochs: 100, Early stopping patience=10

Note on LSTM input shape
────────────────────────
The paper feeds sequences of weekly observations to the LSTM.  Here we use
the last *seq_len* weeks of features as the sequence for each prediction.
"""

from __future__ import annotations

import copy
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
        # Respect the requested dtype instead of hard-coding float32.
        np_dtype = torch.zeros(1, dtype=dtype).numpy().dtype
        return torch.from_numpy(x.astype(np_dtype))
    return torch.tensor(np.asarray(x), dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  DNN Architecture
# ─────────────────────────────────────────────────────────────────────────────

class _DNNNet(nn.Module):
    """Paper DNN: Linear → BatchNorm → ReLU blocks, raw-logit output.

    Note on Softmax
    ---------------
    The paper shows ``Softmax`` in the architecture diagram.  We intentionally
    omit it from the network and apply it only in ``predict_proba``.  This lets
    ``CrossEntropyLoss`` use its internal numerically-stable log-softmax path
    and is mathematically equivalent to the paper's specification.
    """

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
        # keep references to Linear layers for L1 calculation
        self._linear_layers: list[nn.Linear] = []
        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            lin = nn.Linear(in_dim, h)
            self._linear_layers.append(lin)
            layers.append(lin)
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        out_lin = nn.Linear(in_dim, 2)
        self._linear_layers.append(out_lin)
        layers.append(out_lin)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def l1_loss(self) -> torch.Tensor:
        """L1 regularisation term over all weight matrices (not biases)."""
        return self.l1_reg * sum(
            lin.weight.abs().sum() for lin in self._linear_layers
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DNNModel(BaseStockModel):
    """Paper DNN: 3 hidden layers (20-10-5), BatchNorm, L1 reg, RMSprop.

    Training progress is emitted at INFO level every ``log_every_n_epochs``
    epochs and stored in ``self.training_history_`` for post-hoc visualisation.

    Config keys (models.dnn in YAML)
    ---------------------------------
    hidden_layers         : [20, 10, 5]
    epochs                : 100
    early_stopping_patience: 10
    learning_rate         : 0.001
    l1_reg                : 0.0001
    batch_norm            : true
    dropout               : 0.0
    batch_size            : 512
    grad_clip             : 1.0       (max gradient norm; 0 = disabled)
    log_every_n_epochs    : 5
    seed                  : 42
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        mc = cfg._raw["models"]["dnn"]
        self._hidden      = mc.get("hidden_layers", [20, 10, 5])
        self._epochs      = mc.get("epochs", 100)
        self._patience    = mc.get("early_stopping_patience", 10)
        self._lr          = mc.get("learning_rate", 0.001)
        self._l1          = mc.get("l1_reg", 0.0001)
        self._bn          = mc.get("batch_norm", True)
        self._dropout     = mc.get("dropout", 0.0)
        self._batch_size  = mc.get("batch_size", 512)
        self._grad_clip   = mc.get("grad_clip", 1.0)
        self._log_every   = mc.get("log_every_n_epochs", 5)
        self._seed        = mc.get("seed", cfg.seed)
        self._net: _DNNNet | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Training history reset on every fit() call
        self.training_history_: list[dict] = []
        self._window_tag: str = ""         # set by rolling_training for log context

    @property
    def name(self) -> str:
        return "dnn"

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "DNNModel":
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        self.training_history_ = []

        X_np = np.asarray(X_train, dtype=np.float32)
        y_np = np.asarray(y_train, dtype=np.int64)
        n_features = X_np.shape[1]

        self._net = _DNNNet(
            n_features, self._hidden, self._l1, self._bn, self._dropout
        ).to(self._device)

        # Ensure batch_size <= training set size (avoids BatchNorm crash with 1 sample)
        effective_bs = min(self._batch_size, max(2, len(X_np)))
        loader = DataLoader(
            TensorDataset(_to_tensor(X_np), _to_tensor(y_np, dtype=torch.long)),
            batch_size=effective_bs,
            shuffle=True,
            drop_last=(len(X_np) % effective_bs == 1),  # drop lone-sample last batch
        )

        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        if has_val:
            X_v = _to_tensor(np.asarray(X_val, dtype=np.float32)).to(self._device)
            y_v = _to_tensor(np.asarray(y_val, dtype=np.int64), dtype=torch.long).to(self._device)

        opt = torch.optim.RMSprop(self._net.parameters(), lr=self._lr)
        loss_fn = nn.CrossEntropyLoss()
        best_val_loss = np.inf
        best_train_loss = np.inf
        patience_cnt = 0
        best_state: dict | None = None
        n_improvements = 0
        stopped_epoch = self._epochs
        tag = self._window_tag or "dnn"

        logger.info(
            "[%s] Training DNN – %d params, %d train rows, %d val rows, device=%s",
            tag, self._net.param_count(), len(X_np),
            len(X_val) if has_val else 0, self._device,
        )

        for epoch in range(1, self._epochs + 1):
            # ── training pass ──────────────────────────────────────────────
            self._net.train()
            epoch_train_loss = 0.0
            n_batches = 0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                logits = self._net(Xb)
                loss = loss_fn(logits, yb) + self._net.l1_loss()
                loss.backward()
                if self._grad_clip > 0:
                    nn.utils.clip_grad_norm_(self._net.parameters(), self._grad_clip)
                opt.step()
                epoch_train_loss += loss.item()
                n_batches += 1
            epoch_train_loss /= max(n_batches, 1)

            # ── validation pass ────────────────────────────────────────────
            epoch_val_loss: float | None = None
            is_best = False
            if has_val:
                self._net.eval()
                with torch.no_grad():
                    val_logits = self._net(X_v)
                    epoch_val_loss = loss_fn(val_logits, y_v).item()

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_state = copy.deepcopy(self._net.state_dict())
                    patience_cnt = 0
                    n_improvements += 1
                    is_best = True
                else:
                    patience_cnt += 1
            else:
                # No validation: track train loss for best-weight restore
                if epoch_train_loss < best_train_loss:
                    best_train_loss = epoch_train_loss
                    best_state = copy.deepcopy(self._net.state_dict())

            # ── store history ──────────────────────────────────────────────
            self.training_history_.append({
                "epoch":      epoch,
                "train_loss": round(epoch_train_loss, 6),
                "val_loss":   round(epoch_val_loss, 6) if epoch_val_loss is not None else None,
                "is_best":    is_best,
            })

            # ── periodic progress log ──────────────────────────────────────
            if epoch % self._log_every == 0 or epoch == 1:
                val_str = f"  val={epoch_val_loss:.4f}{'*' if is_best else ' '}" if has_val else ""
                logger.info(
                    "[%s] ep %3d/%d  train=%.4f%s",
                    tag, epoch, self._epochs, epoch_train_loss, val_str,
                )

            # ── early stopping ─────────────────────────────────────────────
            if has_val and patience_cnt >= self._patience:
                stopped_epoch = epoch
                logger.info(
                    "[%s] Early stopping at epoch %d  best_val=%.4f  improved %d times",
                    tag, epoch, best_val_loss, n_improvements,
                )
                break

        # ── restore best weights ───────────────────────────────────────────
        if best_state is not None:
            self._net.load_state_dict(best_state)

        logger.info(
            "[%s] Training complete: %d/%d epochs  best_val=%.4f  improvements=%d",
            tag, stopped_epoch, self._epochs,
            best_val_loss if has_val else float("nan"),
            n_improvements,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("DNNModel must be fitted before predict_proba.")
        self._net.eval()
        X_t = _to_tensor(np.asarray(X, dtype=np.float32)).to(self._device)
        with torch.no_grad():
            logits = self._net(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


# ─────────────────────────────────────────────────────────────────────────────
#  LSTM Architecture
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_units: int, dropout: float,
                 l1_reg: float = 0.0) -> None:
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
        self._l1_reg = l1_reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # last time step → (batch, hidden_units)
        return self.fc(self.dropout(last))

    def l1_loss(self) -> torch.Tensor:
        """L1 penalty on all weight tensors (biases excluded)."""
        if self._l1_reg == 0.0:
            return torch.tensor(0.0)
        l1 = sum(p.abs().sum() for name, p in self.named_parameters()
                 if "weight" in name)
        return self._l1_reg * l1

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _build_panel_sequences(
    X: "pd.DataFrame",
    y: "pd.Series | None",
    seq_len: int,
    context: "dict[str, np.ndarray] | None" = None,
) -> "tuple[np.ndarray, np.ndarray | None, np.ndarray]":
    """Build LSTM sequences per ticker from a MultiIndex (date, ticker) DataFrame.

    For each ticker T and each of its N rows in X, a sequence of length
    ``seq_len`` is constructed:

      ``[context[-pad:], X_T[0], ..., X_T[i]]``  (exactly seq_len frames)

    where ``context`` supplies the ``seq_len-1`` most-recent rows from the
    *previous* training window so the first test weeks have full-length inputs.
    When ``context`` is absent (training), the gap is zero-padded.

    Parameters
    ----------
    X:
        MultiIndex DataFrame (date outer, ticker inner) already sorted by date.
        Only the ``values`` array is used; the index is needed for ordering.
    y:
        Labels aligned with X (None during inference).
    seq_len:
        Number of weeks per sequence.
    context:
        ``{ticker: float32 array of shape (k, n_features)}`` where k can be
        anything; only the last ``seq_len-1`` rows are used.

    Returns
    -------
    seqs : float32 (N_seqs, seq_len, n_features)
    labels : int64 (N_seqs,) or None
    positions : int64 (N_seqs,)
        Row offset into X for the sample each sequence predicts, so that
        predictions can be mapped back to the original row order.
    """
    import pandas as pd

    X_sorted = X.sort_index(level="date")
    n_features = X_sorted.shape[1]
    X_vals = X_sorted.values.astype(np.float32)
    y_vals = y.reindex(X_sorted.index).values.astype(np.int64) if y is not None else None

    tickers = X_sorted.index.get_level_values("ticker").unique()

    all_seqs: list[np.ndarray] = []
    all_labels: list[int] = []
    all_positions: list[int] = []

    # Build a mapping: original row index in X (before sort) → sorted position
    # We need positions relative to the *original* X for predict_proba alignment.
    orig_to_sorted = {orig: pos for pos, orig in enumerate(
        X.index.get_indexer(X_sorted.index) if hasattr(X.index, "get_indexer")
        else range(len(X_sorted))
    )}
    # Simpler: track the integer iloc position in X_sorted → the iloc in X
    sorted_to_orig = np.argsort(
        np.argsort(X.index.get_indexer(X_sorted.index))
        if hasattr(X.index, "get_indexer")
        else np.arange(len(X))
    )

    # Build a flat position map: iloc in X_sorted → iloc in X
    # (so predictions can be inserted at the right row of the output array)
    try:
        sorted_index_in_orig = X.index.get_indexer(X_sorted.index)
    except Exception:
        sorted_index_in_orig = np.arange(len(X_sorted))

    for ticker in tickers:
        t_mask = X_sorted.index.get_level_values("ticker") == ticker
        t_sorted_pos = np.where(t_mask)[0]       # rows in X_sorted
        t_orig_pos   = sorted_index_in_orig[t_sorted_pos]  # rows in X

        t_X = X_vals[t_sorted_pos]                # (n_T, n_features)
        n_T = len(t_X)

        # Build prefix: last seq_len-1 rows of prior context (or zeros)
        if context is not None and ticker in context:
            ctx = np.asarray(context[ticker], dtype=np.float32)
            k = min(seq_len - 1, len(ctx))
            prefix = ctx[-k:] if k > 0 else np.empty((0, n_features), dtype=np.float32)
            if k < seq_len - 1:
                extra_pad = np.zeros((seq_len - 1 - k, n_features), dtype=np.float32)
                prefix = np.vstack([extra_pad, prefix])
        else:
            prefix = np.zeros((seq_len - 1, n_features), dtype=np.float32)

        # full_X[i] = X_T[i - (seq_len-1)] for i >= seq_len-1
        full_X = np.vstack([prefix, t_X])         # (seq_len-1 + n_T, n_features)

        for local_i in range(n_T):
            window_start = local_i                 # position in full_X
            window_end   = local_i + seq_len       # exclusive
            seq = full_X[window_start:window_end]  # (seq_len, n_features)
            all_seqs.append(seq)
            if y_vals is not None:
                all_labels.append(int(y_vals[t_sorted_pos[local_i]]))
            all_positions.append(int(t_orig_pos[local_i]))

    seqs      = np.stack(all_seqs, axis=0)         # (N, seq_len, n_features)
    labels    = np.array(all_labels, dtype=np.int64) if all_labels else None
    positions = np.array(all_positions, dtype=np.int64)
    return seqs, labels, positions


class LSTMModel(BaseStockModel):
    """Paper LSTM: 1 hidden layer, 30 cells, seq_len=8 weeks, RMSprop lr=0.001.

    Sequence construction (Section 3.6)
    ─────────────────────────────────────
    Each prediction at week t for ticker T is based on the preceding seq_len
    weekly feature vectors of T.  Sequences are built *per ticker* using a
    sliding window along that ticker's time axis — NOT across tickers.

    At test time, the last ``seq_len-1`` rows of each ticker from the training
    window are stored as a context buffer (``_context``) and prepended to the
    test rows, so the very first test prediction still has a full-length
    input sequence.  Tickers that did not appear in training are zero-padded.

    Training progress is logged at INFO level every ``log_every_n_epochs``
    epochs and stored in ``self.training_history_``.

    Config keys (models.lstm in YAML)
    ----------------------------------
    hidden_units           : 30
    seq_len                : 8
    epochs                 : 100
    early_stopping_patience: 10
    learning_rate          : 0.001
    dropout                : 0.0
    batch_size             : 256
    grad_clip              : 1.0
    log_every_n_epochs     : 5
    seed                   : 42
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        mc = cfg._raw["models"]["lstm"]
        self._hidden_units  = mc.get("hidden_units", 30)
        self._seq_len       = mc.get("seq_len", 8)
        self._epochs        = mc.get("epochs", 100)
        self._patience      = mc.get("early_stopping_patience", 15)
        self._lr            = mc.get("learning_rate", 0.001)
        self._weight_decay  = mc.get("weight_decay", 1e-4)   # L2 reg; matches DNN l1_reg scale
        self._l1_reg        = mc.get("l1_reg", 0.0)          # explicit L1 on weight tensors
        self._dropout       = mc.get("dropout", 0.0)
        self._batch_size    = mc.get("batch_size", 256)
        self._grad_clip     = mc.get("grad_clip", 1.0)
        self._log_every     = mc.get("log_every_n_epochs", 5)
        self._seed          = mc.get("seed", cfg.seed)
        self._net: _LSTMNet | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Stored after fit() for context-aware test-time sequence construction
        self._context: dict[str, np.ndarray] = {}
        # Training history reset on each fit() call
        self.training_history_: list[dict] = []
        self._window_tag: str = ""

    @property
    def name(self) -> str:
        return "lstm"

    # ── internal helpers ─────────────────────────────────────────────────────

    def _has_multiindex(self, X) -> bool:
        import pandas as pd
        return isinstance(X, pd.DataFrame) and isinstance(X.index, pd.MultiIndex)

    def _store_context(self, X) -> None:
        """Store last seq_len-1 rows per ticker from X for test-time prefix."""
        import pandas as pd
        if not self._has_multiindex(X):
            return
        tickers = X.index.get_level_values("ticker").unique()
        self._context = {}
        for t in tickers:
            t_rows = X.xs(t, level="ticker").sort_index()
            tail = t_rows.iloc[-(self._seq_len - 1):]
            self._context[t] = tail.values.astype(np.float32)

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "LSTMModel":
        import pandas as pd
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        self.training_history_ = []

        n_features = np.asarray(X_train).shape[1]
        self._net = _LSTMNet(n_features, self._hidden_units, self._dropout,
                             l1_reg=self._l1_reg).to(self._device)

        tag = self._window_tag or "lstm"

        # ── Build sequences ───────────────────────────────────────────────
        if self._has_multiindex(X_train):
            y_ser = (
                y_train if isinstance(y_train, pd.Series)
                else pd.Series(y_train, index=X_train.index)
            )
            seqs_all, labels_all, pos_all = _build_panel_sequences(
                X_train, y_ser, self._seq_len, context=None
            )

            # Split train vs val by matching val index rows
            if (X_val is not None and y_val is not None
                    and self._has_multiindex(X_val) and len(X_val) > 0):
                # Build an index-based membership set for val rows
                val_index_set = set(map(tuple, X_val.index.tolist()))
                train_index = X_train.index

                is_val_pos = np.array([
                    tuple(train_index[p]) in val_index_set for p in pos_all
                ])
                is_train_pos = ~is_val_pos

                seqs_t   = seqs_all[is_train_pos]
                labels_t = labels_all[is_train_pos]
                seqs_v   = seqs_all[is_val_pos]
                labels_v = labels_all[is_val_pos]
                has_val  = len(seqs_v) > 0
            else:
                seqs_t, labels_t = seqs_all, labels_all
                seqs_v, labels_v = None, None
                has_val = False

            # Store context for test-time sequence construction
            self._store_context(X_train)

        else:
            # Fallback: naive sliding window (no ticker grouping available)
            logger.warning(
                "[%s] X_train has no MultiIndex – using naive sliding-window "
                "sequences (cross-ticker contamination possible).", tag
            )
            X_np = np.asarray(X_train, dtype=np.float32)
            y_np = np.asarray(y_train, dtype=np.int64)
            n, f = X_np.shape
            if n < self._seq_len:
                pad = np.zeros((self._seq_len - n, f), dtype=np.float32)
                X_np = np.vstack([pad, X_np])
                n = self._seq_len
            seqs_t = np.stack(
                [X_np[i: i + self._seq_len] for i in range(n - self._seq_len + 1)],
                axis=0
            )
            labels_t = y_np[self._seq_len - 1:]

            has_val = False
            seqs_v = labels_v = None

            if X_val is not None and y_val is not None and len(X_val) > 0:
                X_v_np = np.asarray(X_val, dtype=np.float32)
                y_v_np = np.asarray(y_val, dtype=np.int64)
                nv = len(X_v_np)
                if nv < self._seq_len:
                    pad = np.zeros((self._seq_len - nv, f), dtype=np.float32)
                    X_v_np = np.vstack([pad, X_v_np])
                seqs_v  = np.stack(
                    [X_v_np[i: i + self._seq_len] for i in range(len(X_v_np) - self._seq_len + 1)],
                    axis=0
                )
                labels_v = y_v_np[self._seq_len - 1:]
                has_val  = len(seqs_v) > 0

        # ── DataLoader ────────────────────────────────────────────────────
        effective_bs = min(self._batch_size, max(2, len(seqs_t)))
        loader = DataLoader(
            TensorDataset(_to_tensor(seqs_t), _to_tensor(labels_t, torch.long)),
            batch_size=effective_bs,
            # Shuffle sequences between epochs so each batch sees a diverse
            # mix of tickers and time periods.  The temporal ordering is
            # preserved WITHIN each sequence (the seq_len steps); shuffling
            # only determines which self-contained sequences appear together
            # in a batch.  Without this, all AAPL sequences appear in the
            # same batches followed by all AMZN sequences, etc., which
            # produces highly correlated gradients and causes the model to
            # memorise ticker-specific temporal patterns that don't generalise.
            shuffle=True,
            drop_last=(len(seqs_t) % effective_bs == 1),
        )

        if has_val:
            X_v_t = _to_tensor(seqs_v).to(self._device)
            y_v_t = _to_tensor(labels_v, torch.long).to(self._device)

        opt = torch.optim.RMSprop(
            self._net.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,   # L2 regularisation
        )
        loss_fn = nn.CrossEntropyLoss()
        best_val_loss   = np.inf
        best_train_loss = np.inf
        patience_cnt = 0
        best_state: dict | None = None
        n_improvements = 0
        stopped_epoch  = self._epochs

        logger.info(
            "[%s] Training LSTM – %d params, %d train seqs, %d val seqs, device=%s",
            tag, self._net.param_count(), len(seqs_t),
            len(seqs_v) if seqs_v is not None else 0, self._device,
        )

        for epoch in range(1, self._epochs + 1):
            # ── training pass ─────────────────────────────────────────────
            self._net.train()
            epoch_train_loss = 0.0
            n_batches = 0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                loss = loss_fn(self._net(Xb), yb) + self._net.l1_loss()
                loss.backward()
                if self._grad_clip > 0:
                    nn.utils.clip_grad_norm_(self._net.parameters(), self._grad_clip)
                opt.step()
                epoch_train_loss += loss.item()
                n_batches += 1
            epoch_train_loss /= max(n_batches, 1)

            # ── validation pass ───────────────────────────────────────────
            epoch_val_loss: float | None = None
            is_best = False
            if has_val:
                self._net.eval()
                with torch.no_grad():
                    epoch_val_loss = loss_fn(self._net(X_v_t), y_v_t).item()
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_state = copy.deepcopy(self._net.state_dict())
                    patience_cnt = 0
                    n_improvements += 1
                    is_best = True
                else:
                    patience_cnt += 1
            else:
                if epoch_train_loss < best_train_loss:
                    best_train_loss = epoch_train_loss
                    best_state = copy.deepcopy(self._net.state_dict())

            # ── store history ─────────────────────────────────────────────
            self.training_history_.append({
                "epoch":      epoch,
                "train_loss": round(epoch_train_loss, 6),
                "val_loss":   round(epoch_val_loss, 6) if epoch_val_loss is not None else None,
                "is_best":    is_best,
            })

            # ── periodic log ──────────────────────────────────────────────
            if epoch % self._log_every == 0 or epoch == 1:
                val_str = (
                    f"  val={epoch_val_loss:.4f}{'*' if is_best else ' '}"
                    if has_val else ""
                )
                logger.info(
                    "[%s] ep %3d/%d  train=%.4f%s",
                    tag, epoch, self._epochs, epoch_train_loss, val_str,
                )

            # ── early stopping ────────────────────────────────────────────
            if has_val and patience_cnt >= self._patience:
                stopped_epoch = epoch
                logger.info(
                    "[%s] Early stopping at epoch %d  best_val=%.4f  improved %d times",
                    tag, epoch, best_val_loss, n_improvements,
                )
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)

        logger.info(
            "[%s] Training complete: %d/%d epochs  best_val=%.4f  improvements=%d",
            tag, stopped_epoch, self._epochs,
            best_val_loss if has_val else float("nan"),
            n_improvements,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return (n_rows, 2) probability array in the same row order as X.

        When X is a MultiIndex DataFrame (date, ticker):
          - sequences are built per ticker using the stored training context
          - predictions are re-sorted back to the original row order of X
        When X is a plain array:
          - naive sliding window; first seq_len-1 rows receive prob=0.5
        """
        if self._net is None:
            raise RuntimeError("LSTMModel must be fitted before predict_proba.")
        self._net.eval()

        if self._has_multiindex(X):
            n_rows = len(X)
            seqs, _, positions = _build_panel_sequences(
                X, y=None, seq_len=self._seq_len, context=self._context
            )
            X_t = _to_tensor(seqs).to(self._device)
            with torch.no_grad():
                logits = self._net(X_t)
                probs_seq = torch.softmax(logits, dim=1).cpu().numpy()

            # Map predictions back to original row order
            output = np.full((n_rows, 2), 0.5, dtype=np.float32)
            output[positions] = probs_seq
            return output

        else:
            # Fallback: naive sliding window
            X_np = np.asarray(X, dtype=np.float32)
            n, f = X_np.shape
            if n < self._seq_len:
                pad = np.zeros((self._seq_len - n, f), dtype=np.float32)
                X_np = np.vstack([pad, X_np])
            seqs = np.stack(
                [X_np[i: i + self._seq_len] for i in range(n - self._seq_len + 1)],
                axis=0
            )
            X_t = _to_tensor(seqs).to(self._device)
            with torch.no_grad():
                logits = self._net(X_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            n_missing = n - len(probs)
            if n_missing > 0:
                probs = np.vstack([np.full((n_missing, 2), 0.5), probs])
            return probs
