"""SBT-1700 RESET — phase-orchestrated pipeline.

Phases:
    manifest       Build the split manifest JSON for a dataset.
    discovery      TRAIN split: exit matrix + per-exit ranker walk-forward.
    validation     VAL split: ≤2 carried exits + ranker eval (model trained on TRAIN).
    final_test     TEST split: ONE-SHOT readout for one locked exit + locked model.
                   Requires `--unlock-test` to bypass the test-period lock.
    report         Synthesize sbt_1700_reset_report.md from prior phase outputs.

Hard rules enforced here:
    - Test split is read only if --unlock-test is passed AND phase=final_test.
    - Discovery never reads validation or test rows.
    - Validation never reads test rows.
    - Final-test model is trained on TRAIN ∪ VAL using a single locked exit
      and locked LightGBM params. Re-tuning is not possible without a code
      change — there are no hyperparameter knobs on the CLI.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sbt1700.exits import EXIT_VARIANTS, simulate_exit, variant_names
from sbt1700.splits import (
    CONTAMINATION_NOTICE,
    PHASES,
    TestLockError,
    load_split,
    write_split_manifest,
)
from sbt1700 import train_ranker, eval_ranker


DEFAULT_DATASET = Path("output/sbt_1700_dataset_reset.parquet")
DEFAULT_MASTER = Path("output/ohlcv_10y_fintables_master.parquet")
DEFAULT_OUT_DIR = Path("output")


# ---------- shared re-simulation ---------------------------------------------

def _by_ticker_ohlc(daily_master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "ticker" not in daily_master.columns:
        raise KeyError("daily master must have a 'ticker' column")
    return {
        tk: g[["Open", "High", "Low", "Close"]].sort_index()
        for tk, g in daily_master.groupby("ticker")
    }


def resimulate_for_variant(
    features_df: pd.DataFrame,
    daily_master: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    """Return features_df augmented with `variant`-specific label columns.

    The original E3 label columns produced by build_dataset are dropped
    so they cannot leak into ranker features. The new label set replaces
    them in place. Rows with no forward bars or invalid ATR are kept but
    the label cells are NaN — downstream `dropna(subset=[label])` removes
    them at fit/eval time.
    """
    if variant not in EXIT_VARIANTS:
        raise KeyError(f"unknown exit variant {variant!r}")
    by_ticker = _by_ticker_ohlc(daily_master)
    label_cols = [
        "exit_variant",
        "realized_R_gross", "realized_R_net", "win_label",
        "tp_hit", "sl_hit", "timeout_hit", "partial_hit",
        "exit_reason", "bars_held",
        "entry_px", "stop_px", "tp_px", "partial_px",
        "atr_1700", "initial_R_price", "exit_px", "exit_date", "cost_R",
    ]
    base = features_df.drop(
        columns=[c for c in label_cols if c in features_df.columns],
        errors="ignore",
    ).copy()
    base["date"] = pd.to_datetime(base["date"])

    out_rows: list[dict] = []
    for r in base.itertuples(index=False):
        sub = by_ticker.get(r.ticker)
        if sub is None or sub.empty:
            continue
        if not hasattr(r, "close_1700") or not hasattr(r, "atr14_prior"):
            # dataset must expose these as features
            continue
        atr = float(r.atr14_prior)
        entry_px = float(r.close_1700)
        sim = simulate_exit(variant, pd.Timestamp(r.date), entry_px, atr, sub)
        merged = {col: getattr(r, col) for col in base.columns}
        merged.update(sim)
        out_rows.append(merged)
    return pd.DataFrame(out_rows)


# ---------- phase: manifest --------------------------------------------------

def phase_manifest(args: argparse.Namespace) -> int:
    out_path = Path(args.out_dir) / "sbt_1700_reset_split_manifest.json"
    manifest = write_split_manifest(args.dataset, out_path)
    print(f"[manifest] wrote {out_path}")
    print(f"[manifest] train     n_rows = {manifest['splits']['train']['n_rows']}")
    print(f"[manifest] validation n_rows = {manifest['splits']['validation']['n_rows']}")
    print(f"[manifest] test       n_rows = (LOCKED)")
    return 0


# ---------- phase: discovery -------------------------------------------------

def phase_discovery(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.dataset, "train")
    if train.empty:
        raise RuntimeError("train split is empty — cannot run discovery")
    daily_master = pd.read_parquet(args.master)
    daily_master.index = pd.to_datetime(daily_master.index)

    cohort_rows: list[dict] = []
    ranker_rows: list[dict] = []
    for variant in variant_names():
        relabeled = resimulate_for_variant(train, daily_master, variant)
        m = eval_ranker.cohort_metrics(relabeled)
        cohort_rows.append({"exit_variant": variant, **asdict(m)})
        if m.n < args.min_n_for_ranker:
            print(f"[discovery] {variant}: N={m.n} < {args.min_n_for_ranker}, skip ranker WF")
            continue
        try:
            wf = train_ranker.walk_forward(relabeled)
            wf.insert(0, "exit_variant", variant)
            ranker_rows.append(wf)
        except ValueError as e:
            print(f"[discovery] {variant}: WF failed ({e})")

    cohort_df = pd.DataFrame(cohort_rows)
    cohort_df.to_csv(out_dir / "sbt_1700_exit_discovery_train.csv", index=False)
    print(f"[discovery] wrote {out_dir / 'sbt_1700_exit_discovery_train.csv'}")
    if ranker_rows:
        ranker_df = pd.concat(ranker_rows, ignore_index=True)
        ranker_df.to_csv(out_dir / "sbt_1700_ranker_discovery_train.csv", index=False)
        print(f"[discovery] wrote {out_dir / 'sbt_1700_ranker_discovery_train.csv'}")
    return 0


# ---------- phase: validation ------------------------------------------------

def phase_validation(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    carried = [v.strip() for v in args.carry.split(",") if v.strip()]
    if not 1 <= len(carried) <= 2:
        raise SystemExit(
            f"--carry must list 1 or 2 exits (got {carried!r}); "
            "discovery promotes at most 2 candidates."
        )
    for v in carried:
        if v not in EXIT_VARIANTS:
            raise SystemExit(f"unknown exit {v!r}; valid: {variant_names()}")

    train = load_split(args.dataset, "train")
    val = load_split(args.dataset, "validation")
    if train.empty or val.empty:
        raise RuntimeError("train or validation split is empty")
    daily_master = pd.read_parquet(args.master)
    daily_master.index = pd.to_datetime(daily_master.index)

    cohort_rows: list[dict] = []
    eval_rows: list[dict] = []
    for variant in carried:
        train_lab = resimulate_for_variant(train, daily_master, variant)
        val_lab = resimulate_for_variant(val, daily_master, variant)
        m = eval_ranker.cohort_metrics(val_lab)
        cohort_rows.append({"exit_variant": variant, **asdict(m)})
        if m.n < args.min_n_for_ranker:
            print(f"[validation] {variant}: N={m.n} < {args.min_n_for_ranker}, skip ranker eval")
            continue
        artifacts = train_ranker.fit_model(train_lab)
        score = eval_ranker.predict_with_model(
            artifacts.model, val_lab.dropna(subset=["realized_R_net"]),
            artifacts.feature_cols)
        rmet = eval_ranker.rank_metrics(val_lab, score)
        eval_rows.append({"exit_variant": variant, "n_train": artifacts.n_train,
                          "n_val": m.n, **rmet})

    cohort_df = pd.DataFrame(cohort_rows)
    cohort_df.to_csv(out_dir / "sbt_1700_exit_validation.csv", index=False)
    print(f"[validation] wrote {out_dir / 'sbt_1700_exit_validation.csv'}")
    if eval_rows:
        eval_df = pd.DataFrame(eval_rows)
        eval_df.to_csv(out_dir / "sbt_1700_ranker_validation.csv", index=False)
        print(f"[validation] wrote {out_dir / 'sbt_1700_ranker_validation.csv'}")
    return 0


# ---------- phase: final_test ------------------------------------------------

def phase_final_test(args: argparse.Namespace) -> int:
    if not args.unlock_test:
        raise TestLockError(
            "phase=final_test requires --unlock-test. " + CONTAMINATION_NOTICE
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.exit not in EXIT_VARIANTS:
        raise SystemExit(f"--exit must be one of {variant_names()}")

    train = load_split(args.dataset, "train")
    val = load_split(args.dataset, "validation")
    test = load_split(args.dataset, "test", allow_test=True)
    if test.empty:
        raise RuntimeError("test split is empty — abort")

    daily_master = pd.read_parquet(args.master)
    daily_master.index = pd.to_datetime(daily_master.index)

    train_val = pd.concat([train, val], ignore_index=True)
    train_val_lab = resimulate_for_variant(train_val, daily_master, args.exit)
    test_lab = resimulate_for_variant(test, daily_master, args.exit)

    artifacts = train_ranker.fit_model(train_val_lab)
    train_ranker.save_artifacts(artifacts, out_dir, tag=f"final_{args.exit}")

    test_eval = eval_ranker.evaluate(artifacts.model, test_lab, artifacts.feature_cols)
    score = eval_ranker.predict_with_model(
        artifacts.model, test_lab.dropna(subset=["realized_R_net"]),
        artifacts.feature_cols)

    by_year = eval_ranker.by_year_summary(test_lab)
    concentration = eval_ranker.concentration_summary(test_lab)

    summary = {
        "exit_variant": args.exit,
        "n_train_val": artifacts.n_train,
        "n_test": int(test_eval["n"]),
        "cohort": test_eval["cohort"],
        "ranker": test_eval["ranker"],
        "by_year": by_year.to_dict(orient="records"),
        "concentration_top10": concentration.to_dict(orient="records"),
        "lock_acknowledgement": (
            "Test split was unlocked once for this readout. Any subsequent "
            "rule change requires a fresh test set or a written "
            "re-contamination note."
        ),
        "contamination_notice": CONTAMINATION_NOTICE,
    }
    summary_path = out_dir / "sbt_1700_final_test_LOCKED.csv"
    pd.DataFrame([{
        "exit_variant": args.exit,
        **test_eval["cohort"],
        **{f"rank_{k}": v for k, v in test_eval["ranker"].items()},
        "n_train_val": artifacts.n_train,
        "n_test": test_eval["n"],
    }]).to_csv(summary_path, index=False)
    (out_dir / "sbt_1700_final_test_LOCKED.json").write_text(
        json.dumps(summary, indent=2, default=str))
    eval_ranker.save_trade_list(
        test_lab.dropna(subset=["realized_R_net"]), score,
        out_dir / "sbt_1700_final_test_LOCKED_trades.csv")
    print(f"[final_test] wrote {summary_path}")
    print(f"[final_test] N_test={test_eval['n']}, "
          f"cohort PF={test_eval['cohort']['pf']:.2f}, "
          f"avg_R={test_eval['cohort']['avg_R']:+.3f}, "
          f"top-decile PF={test_eval['ranker'].get('top_decile_PF', float('nan')):.2f}")
    return 0


# ---------- phase: report ----------------------------------------------------

def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    return df.to_markdown(index=False)


def phase_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    paths = {
        "manifest": out_dir / "sbt_1700_reset_split_manifest.json",
        "discovery_cohort": out_dir / "sbt_1700_exit_discovery_train.csv",
        "discovery_ranker": out_dir / "sbt_1700_ranker_discovery_train.csv",
        "validation_cohort": out_dir / "sbt_1700_exit_validation.csv",
        "validation_ranker": out_dir / "sbt_1700_ranker_validation.csv",
        "final_test_summary": out_dir / "sbt_1700_final_test_LOCKED.csv",
        "final_test_json": out_dir / "sbt_1700_final_test_LOCKED.json",
    }
    lines: list[str] = [
        "# SBT-1700 RESET — Methodology Report",
        "",
        f"> {CONTAMINATION_NOTICE}",
        "",
        "## Split manifest",
        "",
    ]
    if paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text())
        lines.append("```json")
        lines.append(json.dumps(manifest, indent=2))
        lines.append("```")
    else:
        lines.append("_(manifest not produced yet — run `manifest` phase)_")
    lines.append("")

    lines += ["## Discovery (train: 2024-01-16 → 2025-06-30)", ""]
    if paths["discovery_cohort"].exists():
        df = pd.read_csv(paths["discovery_cohort"])
        lines += ["**Raw cohort by exit:**", "", _md_table(df), ""]
    if paths["discovery_ranker"].exists():
        df = pd.read_csv(paths["discovery_ranker"])
        lines += ["**Ranker walk-forward (train internal):**", "", _md_table(df), ""]

    lines += ["## Validation (val: 2025-07-01 → 2025-12-31)", ""]
    if paths["validation_cohort"].exists():
        df = pd.read_csv(paths["validation_cohort"])
        lines += ["**Raw cohort by carried exit:**", "", _md_table(df), ""]
    if paths["validation_ranker"].exists():
        df = pd.read_csv(paths["validation_ranker"])
        lines += ["**Ranker eval (TRAIN-fit, VAL-applied):**", "", _md_table(df), ""]

    lines += ["## Final test (test: 2026-01-01 → 2026-04-24, LOCKED)", ""]
    if paths["final_test_summary"].exists():
        df = pd.read_csv(paths["final_test_summary"])
        lines += ["**One-shot readout:**", "", _md_table(df), ""]
        if paths["final_test_json"].exists():
            j = json.loads(paths["final_test_json"].read_text())
            by_year = pd.DataFrame(j.get("by_year", []))
            conc = pd.DataFrame(j.get("concentration_top10", []))
            if not by_year.empty:
                lines += ["**By year:**", "", _md_table(by_year), ""]
            if not conc.empty:
                lines += ["**Top-10 ticker concentration:**", "", _md_table(conc), ""]
    else:
        lines.append("_(final-test phase not run — readout deliberately gated)_")
    lines.append("")

    lines += [
        "## Decision language",
        "",
        "- **Discovery pass** ≠ validated. Discovery may surface candidates only.",
        "- **Validation pass** ≠ production. Validation freezes one exit + one model.",
        "- **Test pass** = paper candidate only. Live deployment requires a separate, "
          "explicit go-decision and a forward paper-traded ledger built after this reset.",
        "",
    ]
    out_md = out_dir / "sbt_1700_reset_report.md"
    out_md.write_text("\n".join(lines))
    print(f"[report] wrote {out_md}")
    return 0


# ---------- CLI --------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SBT-1700 RESET — phase-orchestrated pipeline.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--min-n-for-ranker", type=int, default=40)
    sub = p.add_subparsers(dest="phase", required=True)

    sub.add_parser("manifest", help="Write the split manifest JSON.")
    sub.add_parser("discovery", help="Discovery on TRAIN: exit matrix + ranker WF.")

    val_p = sub.add_parser("validation",
                           help="Validation on VAL: ≤2 carried exits.")
    val_p.add_argument("--carry", required=True,
                       help="Comma-separated exit variants (1 or 2).")

    test_p = sub.add_parser("final_test",
                            help="ONE-SHOT readout on TEST. Requires --unlock-test.")
    test_p.add_argument("--exit", required=True,
                        help="Locked exit variant for the test readout.")
    test_p.add_argument("--unlock-test", action="store_true",
                        help="Required to read the test split.")

    sub.add_parser("report", help="Synthesize the markdown report.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "manifest":
        return phase_manifest(args)
    if args.phase == "discovery":
        return phase_discovery(args)
    if args.phase == "validation":
        return phase_validation(args)
    if args.phase == "final_test":
        return phase_final_test(args)
    if args.phase == "report":
        return phase_report(args)
    raise SystemExit(f"unknown phase {args.phase!r}; valid: manifest, discovery, "
                     "validation, final_test, report")


if __name__ == "__main__":
    raise SystemExit(main())
