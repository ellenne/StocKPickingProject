"""
run_pipeline_capture.py
Full pipeline runner with complete output capture for paper alignment analysis.
"""
import subprocess
import sys
import os
from datetime import datetime

# ── Ensure outputs dir exists ────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"outputs/full_pipeline_log_{timestamp}.txt"

STEPS = [
    {
        "name": "download-data",
        "cmd": [sys.executable, "-m", "src.cli.main", "--config", "configs/fast.yaml", "-v", "download-data"],
    },
    {
        "name": "build-dataset",
        "cmd": [sys.executable, "-m", "src.cli.main", "--config", "configs/fast.yaml", "-v", "build-dataset"],
    },
    {
        "name": "train-backtest",
        "cmd": [sys.executable, "-m", "src.cli.main", "--config", "configs/fast.yaml", "-v", "train-backtest"],
    },
    {
        "name": "current-picks",
        "cmd": [sys.executable, "-m", "src.cli.main", "--config", "configs/fast.yaml", "-v", "current-picks"],
    },
]


def run_step(step: dict, log_fh) -> int:
    name = step["name"]
    cmd  = step["cmd"]
    sep  = "=" * 60

    print(f"\n{sep}", flush=True)
    print(f"  STEP: {name.upper()}", flush=True)
    print(f"{sep}", flush=True)
    print(f"  Command: {' '.join(cmd)}", flush=True)
    print(flush=True)

    log_fh.write(f"\n{sep}\n")
    log_fh.write(f"STEP: {name.upper()}\n")
    log_fh.write(f"Command: {' '.join(cmd)}\n\n")
    log_fh.flush()

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,          # merge stderr → stdout
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        # Stream output in real time to both console and log file
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fh.write(line)
            log_fh.flush()

        proc.wait()
        rc = proc.returncode

    except Exception as exc:
        msg = f"EXCEPTION running {name}: {exc}\n"
        print(msg, flush=True)
        log_fh.write(msg)
        rc = -1

    status = "SUCCESS" if rc == 0 else f"FAILED (exit {rc})"
    result_line = f"\n[{name}] -> {status}\n"
    print(result_line, flush=True)
    log_fh.write(result_line)
    log_fh.flush()
    return rc


def generate_analysis_summary() -> str:
    """Parse outputs and produce a paper-comparison summary."""
    import pandas as pd

    out = "outputs/paper_analysis_summary.txt"

    with open(out, "w", encoding="utf-8") as f:
        hdr = "=" * 70
        f.write(f"{hdr}\nSTOCK PICKING PROJECT - PAPER ALIGNMENT ANALYSIS\n")
        f.write(f"Wolff & Echterling (2022)  |  Generated {datetime.now():%Y-%m-%d %H:%M:%S}\n{hdr}\n\n")

        # ── 1. Model performance ──────────────────────────────────────────
        f.write("1. MODEL PERFORMANCE RESULTS\n" + "-" * 40 + "\n")
        perf_file = "outputs/metrics_formatted.csv"
        raw_file  = "outputs/metrics.csv"
        perf = None

        for candidate in (perf_file, raw_file):
            if os.path.exists(candidate):
                perf = pd.read_csv(candidate)
                f.write(f"Source: {candidate}\n\n")
                f.write(perf.to_string(index=False))
                f.write("\n\n")
                break

        if perf is None:
            f.write("  [metrics CSV not found - run train-backtest first]\n\n")

        # ── 2. Paper benchmark comparison ────────────────────────────────
        f.write("2. PAPER BENCHMARK COMPARISON\n" + "-" * 40 + "\n")
        paper = {
            "dnn":      {"ann_return_pct": 17.2, "sharpe": 0.67, "capm_alpha_pct": 9.0},
            "lstm":     {"ann_return_pct": 18.1, "sharpe": 0.71, "capm_alpha_pct": 10.0},
            "ensemble": {"ann_return_pct": 20.8, "sharpe": 0.84, "capm_alpha_pct": 12.0},
            "ridge":    {"ann_return_pct": 19.4, "sharpe": 0.77, "capm_alpha_pct": 11.0},
        }

        if perf is not None:
            model_col = next((c for c in perf.columns if "model" in c.lower()), perf.columns[0])
            for model, bench in paper.items():
                row = perf[perf[model_col].str.lower() == model]
                if row.empty:
                    f.write(f"  {model.upper()}: not found in results\n")
                    continue

                row = row.iloc[0]
                f.write(f"\n  {model.upper()}\n")

                for metric_key, paper_val in bench.items():
                    # Try common column name variants
                    col = next(
                        (c for c in perf.columns
                         if any(k in c.lower() for k in metric_key.split("_"))),
                        None,
                    )
                    actual = f"{row[col]}" if col else "N/A"
                    f.write(f"    {metric_key:22s} actual={actual:>10}   paper={paper_val}\n")
        else:
            f.write("  [skip - no metrics file]\n")
        f.write("\n")

        # ── 3. Feature panel ─────────────────────────────────────────────
        f.write("3. DATA / FEATURE PANEL SUMMARY\n" + "-" * 40 + "\n")
        panel_path = "data/processed/features_panel.parquet"
        if os.path.exists(panel_path):
            df = pd.read_parquet(panel_path)
            idx_names = list(df.index.names)
            date_level = next((n for n in idx_names if "date" in str(n).lower()), idx_names[0])
            dates = df.index.get_level_values(date_level)
            missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
            f.write(f"  Shape      : {df.shape}\n")
            f.write(f"  Date range : {dates.min()} → {dates.max()}\n")
            f.write(f"  Missing    : {missing_pct:.2f}%\n\n")
            f.write("  Columns (first 30):\n")
            for c in df.columns[:30]:
                f.write(f"    {c}\n")
        else:
            f.write("  [features_panel.parquet not found]\n\n")

        # ── 4. Equity-curve snapshot ──────────────────────────────────────
        f.write("\n4. EQUITY CURVE SNAPSHOT (last 10 rows)\n" + "-" * 40 + "\n")
        eq_path = "outputs/equity_curves.csv"
        if os.path.exists(eq_path):
            eq = pd.read_csv(eq_path)
            f.write(eq.tail(10).to_string(index=False))
            f.write("\n\n")
        else:
            f.write("  [equity_curves.csv not found]\n\n")

        # ── 5. Current picks ──────────────────────────────────────────────
        f.write("5. LATEST STOCK PICKS (top 20)\n" + "-" * 40 + "\n")
        picks_csv = "outputs/current_picks.csv"
        if os.path.exists(picks_csv):
            picks = pd.read_csv(picks_csv)
            f.write(picks.head(20).to_string(index=False))
            f.write("\n\n")
        else:
            f.write("  [current_picks.csv not found]\n\n")

        # ── 6. Output files inventory ────────────────────────────────────
        f.write("6. OUTPUT FILES INVENTORY\n" + "-" * 40 + "\n")
        if os.path.exists("outputs"):
            files = sorted(os.listdir("outputs"))
            for fname in files:
                fpath = os.path.join("outputs", fname)
                size  = os.path.getsize(fpath)
                f.write(f"  {fname:<45} {size:>10,} bytes\n")

    return out


def create_analysis_package(summary_file: str) -> str:
    """Combine latest pipeline log + summary into one shareable file."""
    pkg = "outputs/COMPLETE_ANALYSIS_PACKAGE.txt"

    with open(pkg, "w", encoding="utf-8") as pkg_f:
        hdr = "=" * 80
        pkg_f.write(f"{hdr}\nSTOCK PICKING PROJECT - COMPLETE ANALYSIS PACKAGE\n")
        pkg_f.write("For Paper Alignment Validation (Wolff & Echterling 2022)\n")
        pkg_f.write(f"Assembled: {datetime.now():%Y-%m-%d %H:%M:%S}\n{hdr}\n\n")

        # Latest pipeline log
        logs = sorted(f for f in os.listdir("outputs") if f.startswith("full_pipeline_log_"))
        if logs:
            pkg_f.write("SECTION 1: COMPLETE PIPELINE EXECUTION LOG\n" + "=" * 50 + "\n")
            with open(f"outputs/{logs[-1]}", encoding="utf-8", errors="replace") as lf:
                pkg_f.write(lf.read())
            pkg_f.write("\n\n")

        # Analysis summary
        if os.path.exists(summary_file):
            pkg_f.write("SECTION 2: PAPER ALIGNMENT ANALYSIS\n" + "=" * 50 + "\n")
            with open(summary_file, encoding="utf-8", errors="replace") as sf:
                pkg_f.write(sf.read())
            pkg_f.write("\n\n")

        # Validation results (if they exist)
        if os.path.exists("tests/validation_results.txt"):
            pkg_f.write("SECTION 3: VALIDATION TEST RESULTS\n" + "=" * 50 + "\n")
            with open("tests/validation_results.txt", encoding="utf-8") as vf:
                pkg_f.write(vf.read())

    return pkg


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("STOCK PICKING PROJECT - COMPLETE PIPELINE RUN WITH OUTPUT CAPTURE")
    print(f"Started : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Log file: {log_file}")
    print("=" * 80, flush=True)

    overall_ok = True

    with open(log_file, "w", encoding="utf-8", errors="replace") as lf:
        lf.write("=" * 80 + "\n")
        lf.write("STOCK PICKING PROJECT - COMPLETE PIPELINE RUN\n")
        lf.write(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        lf.write("=" * 80 + "\n\n")

        for i, step in enumerate(STEPS, 1):
            print(f"\n>>> Executing step {i}/{len(STEPS)}: {step['name']}", flush=True)
            rc = run_step(step, lf)
            if rc != 0:
                overall_ok = False
                lf.write(f"\n[PIPELINE HALTED after {step['name']} failed]\n")
                print(f"\n[PIPELINE HALTED after {step['name']} failed]", flush=True)
                break

        lf.write("\n" + "=" * 80 + "\n")
        lf.write(f"Pipeline {'COMPLETED OK' if overall_ok else 'COMPLETED WITH ERRORS'}\n")
        lf.write(f"Finished: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        lf.write("=" * 80 + "\n")

    print(f"\n{'='*80}")
    print(f"Pipeline {'COMPLETED OK' if overall_ok else 'COMPLETED WITH ERRORS'}")

    # Always try to generate summaries even if a step failed
    print("\nGenerating analysis summary ...", flush=True)
    summary_file = generate_analysis_summary()
    print(f"  → {summary_file}")

    print("\nAssembling complete analysis package ...", flush=True)
    pkg_file = create_analysis_package(summary_file)
    print(f"  → {pkg_file}")

    print(f"\n{'='*80}")
    print("ALL DONE - files written to outputs/")
    print(f"  Pipeline log : {log_file}")
    print(f"  Summary      : {summary_file}")
    print(f"  Full package : {pkg_file}")
    print("=" * 80, flush=True)

    sys.exit(0 if overall_ok else 1)
