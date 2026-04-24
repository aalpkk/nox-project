"""
CTE HB-line tuning sweep (uzman oturumu: cte_hb_v2).

Mirror of tools/fc_tune.py for the HB line. Dataset okunur bir kere, her varyant
aynı split + contract altında eğitilir; artifact yazılmaz (preds/importance
üretilmez), yalnızca özet metrikler tek CSV + konsol tablosu olarak verilir.

Varyant eksenleri:
  - target:   runner_10 / runner_15 / runner_20
  - features: FEATURES_V1 tam | HB-zero_imp drop | HB-top-K subset (HB mean gain)
  - LGBM:    min_child_samples, num_leaves, learning_rate, feature_fraction
  - mode:    "mixed" (HB default) / "pure" (v2b diyagnostik: HB pure 0.91x ≪ mixed 1.14x)

Kapsam (feedback_cte_session_split'e göre):
  - cte/features.py, cte/dataset.py, cte/trigger.py, cte/labels.py, cte/split.py,
    cte/line.py DOKUNULMAZ.
  - FEATURES_V1 listesi DEĞİŞTİRİLMEZ; subset sadece lokal seçim (features arg).
  - LGBMParams defaults train.py içinde dursun; override lokal olarak yapılır.
  - Contract (HB hash) altında verify edilir; mismatch → DUR.
  - relax_m / bar filter gevşetme YAPILMAZ — negatif bulgu (HB lift -34%),
    bkz. memory/cte_trigger_relax_finding.md.

Çıktı:
  output/cte_hb_tune_sweep.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from cte.config import CONFIG
from cte.contract import CONTRACT_PATH, verify_contract
from cte.features import FEATURES_V1
from cte.line import eval_line, train_line
from cte.train import LGBMParams


# ─────────────────────────────────────────────────────────────────────────────
# HB-specific zero/low-gain features (from output/cte_hb_importance_v1.csv)
# ─────────────────────────────────────────────────────────────────────────────
# Strict zero (mean gain == 0 across all 3 folds):
HB_ZERO_IMP = (
    "hb_failed_break_count",
    "hb_prior_break_count",
    "setup_type_code",
)

# Near-zero (mean gain < 3): add these in a tighter prune variant
HB_NEAR_ZERO_IMP = HB_ZERO_IMP + (
    "hb_last_break_age",        # mean 0.95
    "xu100_trend_score_today",  # mean 2.17
)

# Top-15 HB features by mean gain (ordered; from cte_hb_importance_v1.csv)
HB_TOP15 = (
    "dryup_ratio_3_20",
    "bar_body_pct_range",
    "rs_20",
    "breakout_vol_ratio",
    "bar_return_1d",
    "fc_width_cv",
    "fc_convergence",
    "hb_width_atr",
    "rs_5",
    "hb_close_density_atr",
    "dryup_ratio_5_30",
    "compression_score",
    "rs_10",
    "hb_slope_atr",
    "fc_prior_break_count",
)

HB_TOP20 = HB_TOP15 + (
    "bar_close_loc",
    "fc_lower_high_count",
    "fc_width_atr",
    "bb_width_pctile",
    "fc_upper_slope_atr",
)

HB_TOP25 = HB_TOP20 + (
    "breakout_vol_pctile",
    "fc_lower_slope_atr",
    "above_ma50",
    "fc_last_break_age",
    "above_ma20",
)


@dataclass
class Variant:
    name: str
    target: str = "runner_15"
    mode: str = "mixed"                        # HB default
    features: tuple[str, ...] | None = None
    lgbm_overrides: dict = field(default_factory=dict)
    note: str = ""


DEFAULT_VARIANTS: tuple[Variant, ...] = (
    # ── Baseline guard — should reproduce lift@10 1.14x, rho +0.120 ──
    Variant(name="baseline", note="v1 baseline reproduce (mixed, runner_15, 35f)"),

    # ── Horizon sweep ──
    Variant(name="h10", target="runner_10", note="shorter horizon"),
    Variant(name="h20", target="runner_20", note="longer horizon"),

    # ── Feature pruning (HB-specific zero-imp list) ──
    Variant(name="drop_zero_imp",
            features=tuple(c for c in FEATURES_V1 if c not in HB_ZERO_IMP),
            note="drop 3 strict-zero-gain feats"),
    Variant(name="drop_near_zero",
            features=tuple(c for c in FEATURES_V1 if c not in HB_NEAR_ZERO_IMP),
            note="drop 5 near-zero-gain feats"),
    Variant(name="top25", features=HB_TOP25, note="top-25 by HB mean gain"),
    Variant(name="top20", features=HB_TOP20, note="top-20 by HB mean gain"),
    Variant(name="top15", features=HB_TOP15, note="top-15 by HB mean gain"),

    # ── LGBM micro-sweep (HB-local override, train.py defaults unchanged) ──
    Variant(name="mcs10", lgbm_overrides={"min_child_samples": 10},
            note="lower min_child (smaller leaf floor)"),
    Variant(name="mcs30", lgbm_overrides={"min_child_samples": 30},
            note="higher min_child (FC's winner; HB probe)"),
    Variant(name="mcs40", lgbm_overrides={"min_child_samples": 40},
            note="probe mcs=40"),
    Variant(name="leaves15", lgbm_overrides={"num_leaves": 15},
            note="shallower trees"),
    Variant(name="leaves47", lgbm_overrides={"num_leaves": 47},
            note="deeper trees"),
    Variant(name="lr03", lgbm_overrides={"learning_rate": 0.03, "n_estimators": 3000},
            note="slower learn"),
    Variant(name="ff70", lgbm_overrides={"feature_fraction": 0.70},
            note="more feature subsampling"),
    Variant(name="ff100", lgbm_overrides={"feature_fraction": 1.0},
            note="no feature subsampling"),

    # ── Diagnostic: confirm mixed > pure on HB (v2b ablation replay) ──
    Variant(name="pure", mode="pure",
            note="diag: HB pure (expected 0.91x, confirm mixed wins)"),

    # ── Round-2 combos: anchor on best single knobs, layer features ──
    Variant(name="mcs30_top20", features=HB_TOP20,
            lgbm_overrides={"min_child_samples": 30},
            note="mcs30 + top20 feats"),
    Variant(name="mcs30_top15", features=HB_TOP15,
            lgbm_overrides={"min_child_samples": 30},
            note="mcs30 + top15 feats"),
    Variant(name="mcs30_drop_zero",
            features=tuple(c for c in FEATURES_V1 if c not in HB_ZERO_IMP),
            lgbm_overrides={"min_child_samples": 30},
            note="mcs30 + drop zero-gain"),
    Variant(name="mcs30_drop_near_zero",
            features=tuple(c for c in FEATURES_V1 if c not in HB_NEAR_ZERO_IMP),
            lgbm_overrides={"min_child_samples": 30},
            note="mcs30 + drop near-zero"),
    Variant(name="leaves15_drop_zero",
            features=tuple(c for c in FEATURES_V1 if c not in HB_ZERO_IMP),
            lgbm_overrides={"num_leaves": 15},
            note="leaves15 + drop zero"),

    # ── Seed stability — run baseline + best candidates over alt seeds ──
    Variant(name="baseline_seed23", lgbm_overrides={"seed": 23},
            note="baseline seed=23"),
    Variant(name="baseline_seed42", lgbm_overrides={"seed": 42},
            note="baseline seed=42"),
    Variant(name="baseline_seed101", lgbm_overrides={"seed": 101},
            note="baseline seed=101"),
    Variant(name="mcs30_seed23", lgbm_overrides={"min_child_samples": 30, "seed": 23},
            note="mcs30 seed=23"),
    Variant(name="mcs30_seed42", lgbm_overrides={"min_child_samples": 30, "seed": 42},
            note="mcs30 seed=42"),
    Variant(name="mcs30_seed101", lgbm_overrides={"min_child_samples": 30, "seed": 101},
            note="mcs30 seed=101"),
)


def _params_with_overrides(overrides: dict) -> LGBMParams:
    base = asdict(LGBMParams())
    base.update(overrides)
    return LGBMParams(**base)


def run_variant(df: pd.DataFrame, v: Variant, split) -> dict:
    feats_src = list(v.features) if v.features else list(FEATURES_V1)
    feats = [c for c in feats_src if c in df.columns]
    missing = [c for c in feats_src if c not in df.columns]
    params = _params_with_overrides(v.lgbm_overrides)
    print(f"\n▶ variant={v.name}  target={v.target}  mode={v.mode}  "
          f"n_feats={len(feats)}  lgbm_overrides={v.lgbm_overrides or '{}'}")
    if missing:
        print(f"  (dropped missing feats: {missing})")
    res = train_line(df=df, line="hb", mode=v.mode, target=v.target,
                     feature_cols=feats, split=split, params=params)
    evald = eval_line(res.preds, target=v.target, line="hb")
    return {"variant": v, "eval": evald, "imp": res.importance}


def _g(d, *path, default=float("nan")):
    cur = d
    for k in path:
        if cur is None or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarize(row: dict) -> dict:
    v: Variant = row["variant"]
    e = row["eval"]
    over = e.get("overall", {})
    m = over.get("scores", {}).get("score_model", {})
    pf_o = over.get("pf_proxy_overall", {})
    pf_t = over.get("pf_proxy_top30", {})
    deciles = e.get("deciles") or []
    decile_rates = [d["rate"] for d in deciles]
    decile_top = deciles[-1]["rate"] if deciles else float("nan")
    decile_bot = deciles[0]["rate"] if deciles else float("nan")
    byf = e.get("by_fold", {})
    lift_by_fold = {f: _g(byf, f, "scores", "score_model", "lift@10")
                    for f in ("fold1", "fold2", "fold3")}
    rho_by_fold = {f: _g(byf, f, "scores", "score_model", "rho")
                   for f in ("fold1", "fold2", "fold3")}
    bs = e.get("by_setup", {}).get("hb", {})
    lift_setup_hb = _g(bs, "scores", "score_model", "lift@10")
    return {
        "variant": v.name,
        "target": v.target,
        "mode": v.mode,
        "note": v.note,
        "n_overall": over.get("n", 0),
        "base_rate": over.get("base", float("nan")),
        "rho": m.get("rho", float("nan")),
        "p@10": m.get("p@10", float("nan")),
        "p@20": m.get("p@20", float("nan")),
        "lift@10": m.get("lift@10", float("nan")),
        "lift@20": m.get("lift@20", float("nan")),
        "lift@10_fold1": lift_by_fold["fold1"],
        "lift@10_fold2": lift_by_fold["fold2"],
        "lift@10_fold3": lift_by_fold["fold3"],
        "rho_fold1": rho_by_fold["fold1"],
        "rho_fold2": rho_by_fold["fold2"],
        "rho_fold3": rho_by_fold["fold3"],
        "lift@10_setuphb": lift_setup_hb,
        "PF_overall": pf_o.get("PF_proxy", float("nan")),
        "WR_overall": pf_o.get("WR_proxy", float("nan")),
        "PF_top30": pf_t.get("PF_proxy", float("nan")),
        "WR_top30": pf_t.get("WR_proxy", float("nan")),
        "decile_q0": decile_bot,
        "decile_q9": decile_top,
        "decile_spread": (decile_top - decile_bot) if decile_rates else float("nan"),
        "n_features": len(v.features) if v.features else len(FEATURES_V1),
        "lgbm_overrides": json.dumps(v.lgbm_overrides, sort_keys=True),
    }


def _print_table(rows: list[dict]) -> None:
    cols = ["variant", "target", "mode", "n_overall", "base_rate",
            "rho", "p@10", "lift@10",
            "lift@10_fold1", "lift@10_fold2", "lift@10_fold3",
            "lift@10_setuphb", "PF_top30", "WR_top30",
            "decile_q0", "decile_q9",
            "n_features", "note"]
    df = pd.DataFrame(rows)[cols]
    for c in ["base_rate", "rho", "p@10"] + [c for c in df.columns if c.startswith("decile_")] + ["WR_top30"]:
        df[c] = df[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    for c in [c for c in df.columns if c.startswith("lift@") or c.startswith("PF_")]:
        df[c] = df[c].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
    print("\n═══ HB tune sweep summary ═══")
    print(df.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/cte_dataset_v1.parquet")
    ap.add_argument("--out-csv", default="output/cte_hb_tune_sweep.csv")
    ap.add_argument("--ignore-contract", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only run variants with these names (space-separated)")
    args = ap.parse_args()

    os.chdir(str(_ROOT))

    if not Path(args.dataset).exists():
        print(f"❌ Dataset yok: {args.dataset}")
        return 2

    if args.ignore_contract:
        print("⚠ --ignore-contract: HB tune contract atladı")
    else:
        verify_contract(
            dataset_path=args.dataset,
            contract_path=str(CONTRACT_PATH),
            raise_on_mismatch=True,
            verbose=True,
            line="hb",
        )

    df = pd.read_parquet(args.dataset)
    df["date"] = pd.to_datetime(df["date"])
    print(f"[DATA] {args.dataset}  shape={df.shape}")

    variants = DEFAULT_VARIANTS
    if args.only:
        only_set = set(args.only)
        variants = tuple(v for v in variants if v.name in only_set)
        if not variants:
            print(f"❌ No variants match --only {args.only}")
            return 3

    results = []
    for v in variants:
        try:
            row = run_variant(df, v, CONFIG.split)
            results.append(summarize(row))
        except Exception as e:
            print(f"  ✗ variant {v.name} failed: {e}")
            results.append({"variant": v.name, "target": v.target, "mode": v.mode,
                            "note": f"FAILED: {e}"})

    rows_df = pd.DataFrame(results)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(args.out_csv, index=False)
    print(f"\n[WRITE] {args.out_csv}  n_variants={len(results)}")
    _print_table(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
