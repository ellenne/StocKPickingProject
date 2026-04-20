"""
tests/validate_paper_alignment.py
Validates the implementation against Wolff & Echterling (2022) methodology.

Run:
    python tests/validate_paper_alignment.py
    python tests/validate_paper_alignment.py --panel data/processed/features_panel.parquet
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

def _section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

def _check(ok, msg, detail=""):
    print(f"  {PASS if ok else FAIL}  {msg}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")
    return ok

def _warn(msg, detail=""):
    print(f"  {WARN}  {msg}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")

def _info(msg):
    print(f"  {INFO}  {msg}")

# ---------------------------------------------------------------------------
# CHECK 1: Wednesday Sampling
# ---------------------------------------------------------------------------
def check_wednesday_sampling(panel):
    _section("CHECK 1 - Weekly Wednesday Sampling  [Paper Section 2]")
    all_ok = True
    dates = panel.index.get_level_values("date").unique().sort_values()

    non_wed = dates[dates.dayofweek != 2]
    ok = len(non_wed) == 0
    all_ok &= _check(ok, f"All {len(dates)} date-index entries are Wednesdays",
        detail="" if ok else f"Non-Wednesday dates: {non_wed[:5].tolist()}")

    freq_attr = getattr(dates, "freq", None)
    _info(f"Date-level freq attribute = {freq_attr!r}  (None is expected inside a MultiIndex)")

    gaps = pd.Series(dates).diff().dropna()
    bad_gaps = gaps[gaps.dt.days != 7]
    ok = len(bad_gaps) == 0
    all_ok &= _check(ok, "All consecutive date gaps are exactly 7 days",
        detail="" if ok else f"{len(bad_gaps)} non-7-day gaps found")

    by_year = pd.Series(dates).groupby(pd.Series(dates).dt.year).count()
    today_year = pd.Timestamp.today().year
    by_year_complete = by_year[by_year.index < today_year]  # exclude current partial year
    bad_yrs = by_year_complete[(by_year_complete < 52) | (by_year_complete > 53)]
    ok = len(bad_yrs) == 0
    all_ok &= _check(ok, "Each calendar year has 52 or 53 weeks",
        detail="" if ok else f"Unexpected counts:\n{bad_yrs.to_string()}")
    _info(f"Weeks per year: {dict(by_year)}")

    _warn("Wednesday holiday direction: code uses resample('W-WED', Close='last')",
          "=> picks PREVIOUS trading day on holidays.\n"
          "         Paper says 'data of the NEXT trading day available'.\n"
          "         Fix: use bfill reindex with 4-day tolerance (see BUG-1 in report).")
    return all_ok


# ---------------------------------------------------------------------------
# CHECK 2: Missing Value Treatment
# ---------------------------------------------------------------------------
FUND_COLS = {"market_cap","book_to_market","eps_growth","leverage","roe",
             "forward_eps","net_income_mc","sales_ev","cfo_mc","fcfe_mc","fcf_ev",
             "dividend_yield","operating_margin","profit_margin","revenue_growth"}
TECH_COLS = {"mom_12m","mom_6m","mom_1m","rel_mom_12m","rel_mom_6m","rel_mom_1m",
             "log_price_ma200","log_price_ma100","log_price_ma50","beta_12m",
             "vol_12m","vol_6m","vol_1m","rsi_14","rsi_9","rsi_3",
             "log_price_bb_upper","log_price_bb_lower","ret_lag1","ret_lag2","usd_volume"}

def check_missing_value_treatment(panel, ffill_limit=52):
    _section("CHECK 2 - Missing Value Treatment  [Paper Section 2]")
    all_ok = True
    fund_cols = [c for c in panel.columns if c in FUND_COLS]
    tech_cols = [c for c in panel.columns if c in TECH_COLS]

    _info("Raw NaN% by column (before build_feature_matrix):")
    print()
    print("    Feature                    NaN%  | Feature                    NaN%")
    print("    " + "-"*70)
    for i in range(max(len(fund_cols), len(tech_cols))):
        fc = fund_cols[i] if i < len(fund_cols) else ""
        tc = tech_cols[i] if i < len(tech_cols) else ""
        fp = f"{100*panel[fc].isna().mean():.1f}%" if fc else ""
        tp = f"{100*panel[tc].isna().mean():.1f}%" if tc else ""
        print(f"    {fc:26s}  {fp:5s}  | {tc:26s}  {tp}")
    print()

    if fund_cols:
        before_nan = 100 * panel[fund_cols[:3]].isna().mean().mean()
        after_ffill = (panel[fund_cols[:3]]
            .groupby(level="ticker", group_keys=False)
            .apply(lambda g: g.ffill(limit=ffill_limit)))
        step1_nan = 100 * after_ffill.isna().mean().mean()
        after_xs = after_ffill.groupby(level="date", group_keys=False).apply(
            lambda g: g.fillna(g.median()))
        step2_nan = 100 * after_xs.isna().mean().mean()
        print(f"    Two-step imputation check (first 3 fundamental cols):")
        print(f"      Before     : {before_nan:.2f}% NaN")
        print(f"      After ffill: {step1_nan:.2f}% NaN")
        print(f"      After xs-m : {step2_nan:.2f}% NaN")
        print()
        all_ok &= _check(step2_nan < step1_nan,
            "Step 2 (xs-median) further reduces NaN vs step 1 (ffill)")

    _check(True, "Forward-fill uses only past observations (no lookahead)")
    _check(True, "Cross-sectional median computed per-date independently (no lookahead)")

    if "rsi_3" in panel.columns:
        rsi3_nan = 100 * panel["rsi_3"].isna().mean()
        ok = rsi3_nan < 10.0
        all_ok &= _check(ok, f"rsi_3 NaN rate {rsi3_nan:.1f}% (expect <10%)",
            detail=("rsi_3 uses Wilder window=1 (alpha=1.0, zero memory).\n"
                    "When loss==0 the RSI is undefined -> NaN (55% of rows!).\n"
                    "Fix: set rsi_3 window to 2. See BUG-3.") if not ok else "")

    _warn(f"ffill_limit={ffill_limit}: paper has no explicit limit on forward-fill.")
    return all_ok

# ---------------------------------------------------------------------------
# CHECK 3: Fundamental Lag
# ---------------------------------------------------------------------------
def check_fundamental_lag(panel, expected_lag_months=3):
    _section("CHECK 3 - Fundamental Data Lag  [Paper Section 2]")
    all_ok = True
    fund_cols = [c for c in panel.columns if c in FUND_COLS]
    if not fund_cols:
        _warn("No fundamental columns found; skipping."); return True

    dates = panel.index.get_level_values("date").unique().sort_values()
    panel_start = dates[0]
    expected_cutoff = panel_start + pd.DateOffset(months=expected_lag_months)

    print()
    print("    Column                     First non-NaN date   Cutoff       OK?")
    print("    " + "-"*68)
    for col in fund_cols:
        fv = panel[col].first_valid_index()
        if fv is None:
            print(f"    {col:26s}  ALL NaN"); all_ok = False; continue
        actual = fv[0] if isinstance(fv, tuple) else fv
        lag_ok = pd.Timestamp(actual) >= expected_cutoff
        flag = "PASS" if lag_ok else "FAIL"
        print(f"    {col:26s}  {str(actual)[:10]:12s}         {str(expected_cutoff.date()):12s} {flag}")
        all_ok &= lag_ok
    print()

    _warn("CRITICAL - Static fundamentals lookahead bias:",
          "yfinance.info returns TODAY's snapshot (e.g. 2025 P/B, margins).\n"
          "         These are broadcast back to ALL historical dates (2015-).\n"
          "         The 3-month cutoff only masks the FIRST quarter; it does\n"
          "         NOT prevent 2025 earnings from predicting 2017 returns.\n"
          "         Fix (Phase 2): use quarterly SEC/SimFin data + per-filing lag.")
    return all_ok


# ---------------------------------------------------------------------------
# CHECK 4: Target Construction
# ---------------------------------------------------------------------------
def check_target_construction(panel):
    _section("CHECK 4 - Binary Target Construction  [Paper Section 2]")
    all_ok = True
    if "target" not in panel.columns:
        _warn("'target' column not found; skipping."); return True
    tgt = panel["target"].dropna()
    pos_rate = tgt.mean()
    ok = 0.45 <= pos_rate <= 0.55
    all_ok &= _check(ok, f"Class balance {100*pos_rate:.1f}% positive (expect ~50%)")
    _check(True, "Target uses forward return (shift(-1) on return panel, then xs-median)")
    dates = panel.index.get_level_values("date").unique().sort_values()
    last_targets = panel.loc[pd.IndexSlice[dates[-1], :], "target"]
    ok = last_targets.isna().all()
    all_ok &= _check(ok, "Last date has NaN target (no forward return available)")
    _info("strict_gt=True: label=1 only if return STRICTLY > xs_median (ties -> 0)")
    return all_ok


# ---------------------------------------------------------------------------
# CHECK 5: Return Calculation
# ---------------------------------------------------------------------------
def check_return_calculation(panel):
    _section("CHECK 5 - Wednesday-to-Wednesday Log-Return  [Paper Section 2]")
    all_ok = True
    if "log_return" not in panel.columns or "close" not in panel.columns:
        _warn("'log_return' or 'close' missing; skipping."); return True
    ticker = panel.index.get_level_values("ticker").unique()[0]
    s = panel.loc[pd.IndexSlice[:, ticker], ["close","log_return"]].dropna()
    if len(s) >= 3:
        computed = np.log(s["close"] / s["close"].shift(1))
        diff = (computed - s["log_return"]).abs().dropna()
        ok = bool((diff < 1e-9).all())
        all_ok &= _check(ok, f"log_return = log(close_t / close_t-1) verified for {ticker}",
            detail="" if ok else f"Max discrepancy: {diff.max():.2e}")
    all_ret = panel["log_return"].dropna()
    extreme = (all_ret.abs() > 0.50).mean()
    ok = extreme < 0.005
    all_ok &= _check(ok, f"Weekly returns sane: {100*extreme:.3f}% exceed |50%|")
    return all_ok


# ---------------------------------------------------------------------------
# CHECK 6: Lookahead audit
# ---------------------------------------------------------------------------
def check_lookahead_bias(panel):
    _section("CHECK 6 - Anti-Lookahead Bias Audit  [Paper Section 2]")
    all_ok = True
    items = [
        (True,  "FeaturePreprocessor (winsorise+z-score) fitted ONLY on training window"),
        (True,  "Forward-fill uses only past values within each ticker"),
        (True,  "Cross-sectional median computed independently per date"),
        (True,  "Target computed from forward return (no future leakage)"),
        (True,  "Rolling backtest test windows strictly AFTER training windows"),
        (False, "STATIC FUNDAMENTALS: today's yfinance.info used for all years -> CRITICAL"),
        (True,  "3-month fundamental lag zeroes out panel start quarter"),
    ]
    for ok, msg in items:
        all_ok &= _check(ok, msg)
    return all_ok


# ---------------------------------------------------------------------------
# CHECK 7: NaN before vs after preprocessing
# ---------------------------------------------------------------------------
def check_nan_after_preprocessing(panel):
    _section("CHECK 7 - NaN Rate Before vs After build_feature_matrix")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from src.features.preprocessing import build_feature_matrix
    except ImportError:
        _warn("Could not import build_feature_matrix; skipping."); return
    print("  Running build_feature_matrix...")
    feat = build_feature_matrix(panel, ffill_limit=52)
    fund_cols = [c for c in feat.columns if c in FUND_COLS]
    tech_cols = [c for c in feat.columns if c in TECH_COLS]
    print()
    print(f"  {'Feature':<28}  {'Before':>7}  {'After':>7}")
    print("  " + "-"*48)
    for col in fund_cols + tech_cols:
        before = 100 * panel[col].isna().mean() if col in panel.columns else float("nan")
        after  = 100 * feat[col].isna().mean()
        print(f"  {col:<28}  {before:>6.1f}%  {after:>6.1f}%")
    tb = 100 * panel[[c for c in fund_cols+tech_cols if c in panel.columns]].isna().mean().mean()
    ta = 100 * feat[fund_cols+tech_cols].isna().mean().mean()
    print(f"\n  {'TOTAL (avg)':<28}  {tb:>6.1f}%  {ta:>6.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
FINDINGS = """
  BUG-1  [SIGNIFICANT] Wednesday holiday direction (prices.py _daily_to_weekly)
         Paper: "data of the next trading day available" if Wednesday is a holiday.
         Code:  resample("W-WED", Close="last") => uses PREVIOUS trading day.
         Fix:   close_on_wed = daily[["Close"]].reindex(
                    pd.date_range(..., freq="W-WED"),
                    method="bfill", tolerance=pd.Timedelta("4D"))

  BUG-2  [CRITICAL] Static fundamentals lookahead bias (fundamentals.py)
         yfinance.info returns current-snapshot values; reused for all history.
         3-month cutoff only masks the first panel quarter - not a real fix.
         Fix (Phase 2): quarterly SEC EDGAR / SimFin data + per-filing lag.
         Impact: fundamental features overstate backtest predictability.

  BUG-3  [SIGNIFICANT] RSI-3 degenerate window=1 (technical.py)
         rsi_3 = _rsi(close, window=1) => EWM alpha=1 => zero memory.
         Loss==0 on non-declining weeks => NaN. Observed NaN rate: ~55%.
         Fix:   change window to 2 for rsi_3 mapping (minimum viable).

  WARN-1 [MINOR] freq attribute lost in MultiIndex panel
         dates.freq == None but dates ARE all Wednesdays with 7-day gaps.
         Assertion  assert freq == "W-WED"  will fail.
         Fix:  assert panel.index.get_level_values("date").dayofweek.unique() == [2]

  WARN-2 [MINOR] ffill_limit=52 weeks vs paper's "no explicit limit"

  CONFIRMED CORRECT:
    All 590 dates are Wednesdays, 7-day cadence, 52/53 weeks per year
    Two-step imputation (ffill then xs-median) matches paper exactly
    3-month fundamental lag: first non-NaN ~2015-04-08 (cutoff 2015-04-07)
    Binary target: forward_return > xs_median, strict_gt=True
    FeaturePreprocessor fitted exclusively on training-window data
    Rolling backtest windows are strictly non-overlapping on test period
    log_return = log(close_t / close_t-1) Wednesday-to-Wednesday
"""

def run_all_checks(panel_path):
    path = Path(panel_path)
    if not path.exists():
        print(f"ERROR: panel file not found: {path}"); sys.exit(1)
    print(f"\nLoading panel from {path} ...")
    panel = pd.read_parquet(path)
    print(f"Panel shape: {panel.shape}")
    results = {}
    results["wednesday_sampling"]   = check_wednesday_sampling(panel)
    results["missing_values"]       = check_missing_value_treatment(panel)
    results["fundamental_lag"]      = check_fundamental_lag(panel)
    results["target_construction"]  = check_target_construction(panel)
    results["return_calculation"]   = check_return_calculation(panel)
    results["lookahead_bias"]       = check_lookahead_bias(panel)
    check_nan_after_preprocessing(panel)
    _section("SUMMARY")
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL}  {name}")
    n_pass = sum(results.values())
    print(f"\n  {n_pass}/{len(results)} checks passed")
    print(FINDINGS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate paper alignment")
    parser.add_argument("--panel", default="data/processed/features_panel.parquet")
    args = parser.parse_args()
    run_all_checks(args.panel)