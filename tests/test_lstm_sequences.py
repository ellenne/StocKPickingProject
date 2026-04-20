"""Quick smoke-test for the LSTM sequence builder and LSTMModel."""
import numpy as np
import pandas as pd

from src.config import load_config
from src.models.neural_models import LSTMModel, _build_panel_sequences

SEQ_LEN   = 4
N_TICKERS = 5
N_WEEKS   = 20
N_FEAT    = 8


def make_panel():
    np.random.seed(0)
    dates   = pd.date_range("2020-01-01", periods=N_WEEKS, freq="W-WED")
    tickers = [f"T{i}" for i in range(N_TICKERS)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    X = pd.DataFrame(
        np.random.randn(len(idx), N_FEAT).astype("float32"),
        index=idx,
        columns=[f"f{i}" for i in range(N_FEAT)],
    )
    y = pd.Series(np.random.randint(0, 2, len(idx)), index=idx)
    return X, y, dates


def test_build_panel_sequences():
    X, y, dates = make_panel()
    seqs, labels, pos = _build_panel_sequences(X, y, SEQ_LEN, context=None)

    assert seqs.shape == (N_TICKERS * N_WEEKS, SEQ_LEN, N_FEAT), \
        f"unexpected shape: {seqs.shape}"
    assert labels is not None and len(labels) == N_TICKERS * N_WEEKS
    assert len(pos) == N_TICKERS * N_WEEKS
    assert int(pos.max()) < len(X)

    # First seq of T0 should be zero-padded in the first SEQ_LEN-1 frames
    t0_first_idx = next(
        i for i, p in enumerate(pos) if X.index[p] == (dates[0], "T0")
    )
    first_seq = seqs[t0_first_idx]
    assert np.allclose(first_seq[: SEQ_LEN - 1], 0.0), \
        "First SEQ_LEN-1 frames should be zero-padded"
    print("test_build_panel_sequences: PASSED")


def test_context_buffer():
    """Context from training prefix should replace zero-padding."""
    X, y, dates = make_panel()
    train_dates = dates[:16]
    X_tr = X.loc[X.index.get_level_values("date").isin(train_dates)]
    y_tr = y.loc[X_tr.index]

    # Build context from the tail of training data
    context = {}
    for t in [f"T{i}" for i in range(N_TICKERS)]:
        rows = X_tr.xs(t, level="ticker").sort_index()
        context[t] = rows.iloc[-(SEQ_LEN - 1):].values.astype("float32")

    X_test = X.loc[~X.index.get_level_values("date").isin(train_dates)]
    seqs_ctx, _, pos = _build_panel_sequences(X_test, None, SEQ_LEN, context=context)

    # First test seq for T0 should NOT be all-zero (context came from training)
    t0_first_idx = next(
        i for i, p in enumerate(pos) if X_test.index[p][1] == "T0"
    )
    first_seq = seqs_ctx[t0_first_idx]
    assert not np.allclose(first_seq[: SEQ_LEN - 1], 0.0), \
        "With context provided, first frames should NOT be zero"
    print("test_context_buffer: PASSED")


def test_lstm_fit_predict():
    cfg = load_config("configs/fast.yaml")
    X, y, dates = make_panel()
    train_dates = dates[:16]
    val_dates   = dates[16:]

    X_tr = X.loc[X.index.get_level_values("date").isin(train_dates)]
    y_tr = y.loc[X_tr.index]
    X_v  = X.loc[X.index.get_level_values("date").isin(val_dates)]
    y_v  = y.loc[X_v.index]

    model = LSTMModel(cfg)
    model._window_tag = "lstm-test"
    model.fit(X_tr, y_tr, X_v, y_v)

    assert len(model.training_history_) > 0, "training_history_ must be populated"
    assert len(model._context) == N_TICKERS, "context must have one entry per ticker"

    # Context shape: (seq_len-1, n_features)
    lstm_seq_len = cfg._raw["models"]["lstm"]["seq_len"]
    for t, ctx in model._context.items():
        expected_rows = min(lstm_seq_len - 1, 16)
        assert ctx.shape == (expected_rows, N_FEAT), \
            f"bad context shape for {t}: {ctx.shape}"

    proba = model.predict_proba(X_v)
    assert proba.shape == (len(X_v), 2), f"bad shape: {proba.shape}"
    assert abs(proba.sum(axis=1) - 1.0).max() < 1e-5, "proba rows must sum to 1"

    # Row-order consistency: shuffled input should give same predictions (re-indexed)
    X_shuf = X_v.sample(frac=1, random_state=7)
    p_shuf = model.predict_proba(X_shuf)
    for i, (dt, tk) in enumerate(X_v.index[:15]):
        j = X_shuf.index.get_loc((dt, tk))
        assert np.allclose(proba[i], p_shuf[j], atol=1e-6), \
            f"Row-order mismatch at {(dt, tk)}"

    print("test_lstm_fit_predict: PASSED")


if __name__ == "__main__":
    test_build_panel_sequences()
    test_context_buffer()
    test_lstm_fit_predict()
    print("ALL LSTM TESTS PASSED")
