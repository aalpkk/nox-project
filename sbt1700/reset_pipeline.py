"""SBT-1700 RESET — phase-orchestrated pipeline.

Phases:
    manifest         Build the split manifest JSON for a dataset.
    discovery        TRAIN split: legacy E3..E7 exit matrix + per-exit
                     ranker walk-forward.
    discovery_grid   TRAIN split: F0..F4 controlled-grammar exit grid
                     (~66 variants). Cohort metrics only — no ranker
                     work. Ranker walk-forward is reserved for the ≤3
                     carried variants in the validation phase.
    validation       VAL split: ≤3 carried exits + ranker eval (model
                     trained on TRAIN).
    final_test       TEST split: ONE-SHOT readout for one locked exit +
                     locked model. Requires `--unlock-test` to bypass
                     the test-period lock.
    report           Synthesize sbt_1700_reset_report.md from prior
                     phase outputs.

Hard rules enforced here:
    - Test split is read only if --unlock-test is passed AND phase=final_test.
    - Discovery (legacy and grid) never reads validation or test rows.
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

import numpy as np

from sbt1700.exits import EXIT_VARIANTS, simulate_exit, variant_names
from sbt1700.exits_v2 import simulate_exit_v2
from sbt1700.exit_grid import build_grid_v2, grid_summary, is_v2_name, resolve_exit_spec
from sbt1700.exit_discovery import run_discovery_grid
from sbt1700.capture_decomp import run_capture_decomp
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

    Dispatches by name prefix:
      * legacy E3..E7  → `sbt1700.exits.simulate_exit` (5-field outcome)
      * v2 F0..F8      → `sbt1700.exits_v2.simulate_exit_v2`
        (uses prior-close history for EMA bootstrap and optional
        box_top/box_bottom for structure-based trend exits)

    Both legacy and v2 label columns are stripped from `features_df`
    before re-simulation so they cannot leak into ranker features and
    so v2-only fields don't survive a v1→v2 (or vice-versa) re-pass.
    Rows with no forward bars / invalid ATR are kept but with NaN
    labels — downstream `dropna(subset=[label])` removes them at
    fit/eval time.
    """
    is_v2 = is_v2_name(variant)
    if not is_v2 and variant not in EXIT_VARIANTS:
        raise KeyError(
            f"unknown exit variant {variant!r}; "
            f"legacy: {sorted(EXIT_VARIANTS)}; "
            f"v2: use sbt1700.exit_grid.resolve_exit_spec for diagnostics"
        )
    cfg = resolve_exit_spec(variant) if is_v2 else None
    by_ticker = _by_ticker_ohlc(daily_master)
    # Drop both legacy E3..E7 outcome columns AND v2-specific outcome
    # columns. Re-simulation produces a fresh outcome set; carrying the
    # old set forward would risk shape mismatch and leak.
    label_cols = [
        # shared
        "exit_variant", "realized_R_gross", "realized_R_net", "win_label",
        "exit_reason", "bars_held",
        "entry_px", "stop_px", "atr_1700", "initial_R_price",
        "exit_px", "exit_date", "cost_R",
        "partial_px", "partial_hit",
        # legacy E3..E7 only
        "tp_hit", "sl_hit", "timeout_hit", "tp_px",
        # v2 only
        "exit_family",
        "initial_stop_hit", "breakeven_stop_hit", "profit_lock_stop_hit",
        "partial2_hit", "partial2_px",
        "trend_exit_hit", "max_hold_hit",
        "MFE_R", "giveback_R", "captured_MFE_ratio",
    ]
    base = features_df.drop(
        columns=[c for c in label_cols if c in features_df.columns],
        errors="ignore",
    ).copy()
    base["date"] = pd.to_datetime(base["date"])

    use_box = ("box_top" in base.columns) and ("box_bottom" in base.columns)

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
        entry_date = pd.Timestamp(r.date)
        if is_v2:
            prior_closes = sub.loc[sub.index < entry_date, "Close"].values.astype(float)
            bt = float(getattr(r, "box_top", float("nan"))) if use_box else float("nan")
            bb = float(getattr(r, "box_bottom", float("nan"))) if use_box else float("nan")
            sim = simulate_exit_v2(
                cfg,
                entry_date=entry_date,
                entry_px=entry_px,
                atr_1700=atr,
                forward_ohlc=sub,
                prior_closes=prior_closes,
                box_top=bt if np.isfinite(bt) else None,
                box_bottom=bb if np.isfinite(bb) else None,
            )
        else:
            sim = simulate_exit(variant, entry_date, entry_px, atr, sub)
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


# ---------- phase: discovery_grid --------------------------------------------

def phase_discovery_grid(args: argparse.Namespace) -> int:
    """Train-only F0..F4 exit-discovery grid.

    Hard contract:
        - Reads ONLY the train split (`load_split(..., "train")`).
        - Never touches validation or test rows.
        - Does not train a model; it produces variant-level cohort metrics
          and a max-3 carried-candidate CSV. Ranker work belongs in a
          later phase that consumes these candidates.
        - After validation phase starts, no new family or parameter may
          be added to the grid — bump the suffix and document a new
          methodology note instead.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.dataset, "train")
    if train.empty:
        raise RuntimeError("train split is empty — cannot run discovery_grid")
    daily_master = pd.read_parquet(args.master)
    daily_master.index = pd.to_datetime(daily_master.index)

    grid = build_grid_v2()
    counts = grid_summary()
    print("[discovery_grid] grid:")
    for fam, n in counts.items():
        print(f"  {fam:<28} {n}")
    print(f"[discovery_grid] train rows: {len(train)}")

    summary = run_discovery_grid(
        train_panel=train,
        daily_master=daily_master,
        grid=grid,
        out_dir=out_dir,
    )
    print(f"[discovery_grid] DONE — {summary['n_variants']} variants, "
          f"{summary['n_carried']} carried")
    return 0


# ---------- phase: capture_decomp -------------------------------------------

DEFAULT_CARRIED_FOR_DECOMP = (
    "F0_no_partial_trend_sl1.5_p0_ema10_h40",
    "F4_structure_sl1.5_p50at1R_ema10_h40",
    "F0_no_partial_trend_sl1.5_p0_atr2.0_h20",
)


def phase_capture_decomp(args: argparse.Namespace) -> int:
    """TRAIN-only per-path capture decomposition for carried variants.

    Tags each TRAIN signal with `path_type ∈ {parabolic, spike_fade, clean}`
    via `path_type.classify_panel`, runs the F0/F4 carried variants, and
    writes per-cell aggregates to
    `output/sbt_1700_capture_decomp_train.csv`.

    The classifier is exit-agnostic, so cross-variant capture differences
    on the same path reflect *exit behaviour*, not selection drift.

    Hard contract: never reads validation or test rows. The carried list
    is fixed at the discovery_grid output and is not user-tunable on the
    CLI to keep the methodology auditable.
    """
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.dataset, "train")
    if train.empty:
        raise RuntimeError("train split is empty — cannot run capture_decomp")
    daily_master = pd.read_parquet(args.master)
    daily_master.index = pd.to_datetime(daily_master.index)

    grid = build_grid_v2()
    by_name = {v.name: v for v in grid}
    carried_names = [v.strip() for v in args.carry.split(",") if v.strip()] \
        if args.carry else list(DEFAULT_CARRIED_FOR_DECOMP)
    carried = []
    for nm in carried_names:
        cfg = by_name.get(nm)
        if cfg is None:
            raise SystemExit(f"unknown carried variant {nm!r}")
        carried.append(cfg)

    print(f"[capture_decomp] train rows: {len(train)}")
    print("[capture_decomp] carried:")
    for cfg in carried:
        print(f"  {cfg.name}  (family={cfg.family}, trend={cfg.trend_kind})")

    summary = run_capture_decomp(
        panel=train,
        daily_master=daily_master,
        carried_variants=carried,
        out_dir=out_dir,
        suffix="train",
    )
    print(f"[capture_decomp] wrote {summary['decomp_csv']}")
    print(f"[capture_decomp] wrote {summary['cohort_csv']}")
    print(f"[capture_decomp] wrote {summary['paths_csv']}")
    print(f"[capture_decomp] panel n={summary['n_panel']}, "
          f"unknown_path={summary['n_path_unknown']}, "
          f"variants={summary['n_variants']}")
    return 0


# ---------- phase: validation ------------------------------------------------

def phase_validation(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    carried = [v.strip() for v in args.carry.split(",") if v.strip()]
    if not 1 <= len(carried) <= 4:
        raise SystemExit(
            f"--carry must list 1 to 4 exits (got {carried!r}); "
            "discovery_grid promotes at most 4 candidates "
            "(slot 4 is the conditional F8 profit-lock carry)."
        )
    for v in carried:
        if not (v in EXIT_VARIANTS or is_v2_name(v)):
            raise SystemExit(
                f"unknown exit {v!r}; "
                f"legacy E3..E7: {variant_names()}; "
                f"v2 names come from sbt1700.exit_grid.build_grid_v2() "
                f"(F0..F8 grammar)"
            )

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
    if not (args.exit in EXIT_VARIANTS or is_v2_name(args.exit)):
        raise SystemExit(
            f"--exit unknown {args.exit!r}; "
            f"legacy E3..E7: {variant_names()}; "
            f"v2 names come from sbt1700.exit_grid.build_grid_v2() "
            f"(F0..F8 grammar)"
        )

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
        "discovery_grid": out_dir / "sbt_1700_exit_discovery_grid.csv",
        "discovery_grid_family": out_dir / "sbt_1700_exit_discovery_family_summary.csv",
        "discovery_grid_md": out_dir / "sbt_1700_exit_discovery_recommended_validation_exits.md",
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
    sub.add_parser(
        "discovery_grid",
        help="Train-only F0..F4 exit-discovery grid (~66 variants), "
             "no ranker, no validation/test access.",
    )

    cap_p = sub.add_parser(
        "capture_decomp",
        help="Train-only per-path capture decomposition for carried variants.",
    )
    cap_p.add_argument(
        "--carry", default="",
        help="Optional comma-separated list of variant names to override the "
             "default carried set (3 variants from discovery_grid).",
    )

    val_p = sub.add_parser("validation",
                           help="Validation on VAL: ≤3 carried exits.")
    val_p.add_argument("--carry", required=True,
                       help="Comma-separated exit variants (1 to 3).")

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
    if args.phase == "discovery_grid":
        return phase_discovery_grid(args)
    if args.phase == "capture_decomp":
        return phase_capture_decomp(args)
    if args.phase == "validation":
        return phase_validation(args)
    if args.phase == "final_test":
        return phase_final_test(args)
    if args.phase == "report":
        return phase_report(args)
    raise SystemExit(f"unknown phase {args.phase!r}; valid: manifest, discovery, "
                     "discovery_grid, capture_decomp, validation, final_test, report")


if __name__ == "__main__":
    raise SystemExit(main())
