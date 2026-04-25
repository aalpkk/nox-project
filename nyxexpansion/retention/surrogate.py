"""Production load + predict API for the timing-clean retention surrogate.

The surrogate is a LightGBM stand-in for the in-memory v4C regressor used at
17:00 TR re-rank time. Two heads:

- UP head trained on ``model_kind == "up"`` rows of ``nyxexp_preds_v4C.parquet``
  with ``CORE_FEATURES_UP`` (33 cols, V1 + J + chase_score_soft).
- NONUP head trained on ``model_kind == "nonup"`` rows with ``CORE_FEATURES_V1``
  (26 cols, no J block, no chase).

Artifact format (pickle): a dict produced by ``train_surrogate.py``. The
sidecar JSON next to the pickle is a metadata mirror that can be inspected
without unpickling.

Public API:
- ``load(path)`` → ``RetentionSurrogate``
- ``RetentionSurrogate.predict(feats_df)`` — feats_df must include
  ``model_kind``; returns a Series of surrogate ``winner_R_pred`` aligned to
  feats_df.index.
- ``RetentionSurrogate.predict_up(feats_df)`` /
  ``RetentionSurrogate.predict_nonup(feats_df)`` — direct calls.

Schema discipline (fail-fast):
- Missing columns → ``MissingFeatureError``
- Extra columns are ignored at predict time but logged at load time
- Schema version mismatch between artifact and current code → ``SchemaVersionMismatch``

Bumping the schema version (``TRUNCATED_FEATURE_SCHEMA_VERSION``) requires
retraining and re-persisting.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from nyxexpansion.retention import TRUNCATED_FEATURE_SCHEMA_VERSION


SURROGATE_MODEL_VERSION = "v1"


class SchemaVersionMismatch(RuntimeError):
    """Raised when artifact's schema version does not match the current code."""


class MissingFeatureError(RuntimeError):
    """Raised when an input frame is missing columns the surrogate requires."""


@dataclass
class _Head:
    feature_columns: list[str]
    feature_columns_hash: str
    model: Any
    best_iteration: int | None
    validation_rho: float
    n_train: int
    n_val: int


@dataclass
class RetentionSurrogate:
    model_version: str
    truncated_feature_schema_version: str
    training_dataset_path: str
    training_dataset_hash: str
    created_at: str
    regime_split_rule: str
    lgbm_params: dict
    up: _Head
    nonup: _Head
    artifact_path: Path = field(default_factory=Path)

    def _check_columns(self, head: _Head, df: pd.DataFrame) -> None:
        missing = [c for c in head.feature_columns if c not in df.columns]
        if missing:
            raise MissingFeatureError(
                f"surrogate input is missing {len(missing)} required columns: "
                f"{missing[:8]}{'…' if len(missing) > 8 else ''}"
            )

    def predict_up(self, feats_df: pd.DataFrame) -> pd.Series:
        self._check_columns(self.up, feats_df)
        X = feats_df[self.up.feature_columns].astype(float)
        yhat = self.up.model.predict(
            X, num_iteration=self.up.best_iteration,
        )
        return pd.Series(yhat, index=feats_df.index, name="winner_R_pred_tr")

    def predict_nonup(self, feats_df: pd.DataFrame) -> pd.Series:
        self._check_columns(self.nonup, feats_df)
        X = feats_df[self.nonup.feature_columns].astype(float)
        yhat = self.nonup.model.predict(
            X, num_iteration=self.nonup.best_iteration,
        )
        return pd.Series(yhat, index=feats_df.index, name="winner_R_pred_tr")

    def predict(self, feats_df: pd.DataFrame) -> pd.Series:
        """Score a frame containing both regimes. Routes by ``model_kind``."""
        if "model_kind" not in feats_df.columns:
            raise MissingFeatureError(
                "predict() requires a 'model_kind' column to route UP vs NONUP"
            )
        out = pd.Series(float("nan"), index=feats_df.index,
                        name="winner_R_pred_tr")
        up_mask = feats_df["model_kind"] == "up"
        nu_mask = feats_df["model_kind"] == "nonup"
        if up_mask.any():
            out.loc[up_mask] = self.predict_up(feats_df.loc[up_mask])
        if nu_mask.any():
            out.loc[nu_mask] = self.predict_nonup(feats_df.loc[nu_mask])
        return out


def load(path: str | Path) -> RetentionSurrogate:
    """Load a persisted surrogate artifact and validate its schema version."""
    p = Path(path)
    with p.open("rb") as fh:
        bundle = pickle.load(fh)

    artifact_schema = bundle.get("truncated_feature_schema_version")
    if artifact_schema != TRUNCATED_FEATURE_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"artifact schema {artifact_schema!r} != code schema "
            f"{TRUNCATED_FEATURE_SCHEMA_VERSION!r} — retrain the surrogate"
        )

    surrogate = RetentionSurrogate(
        model_version=bundle["model_version"],
        truncated_feature_schema_version=artifact_schema,
        training_dataset_path=bundle["training_dataset_path"],
        training_dataset_hash=bundle["training_dataset_hash"],
        created_at=bundle["created_at"],
        regime_split_rule=bundle["regime_split_rule"],
        lgbm_params=bundle["lgbm_params"],
        up=_Head(**bundle["up"]),
        nonup=_Head(**bundle["nonup"]),
        artifact_path=p,
    )
    return surrogate


def sidecar_metadata_path(artifact_path: str | Path) -> Path:
    p = Path(artifact_path)
    return p.with_suffix(".json")


def write_sidecar(artifact_path: str | Path, bundle: dict) -> Path:
    """Mirror the non-binary fields of the artifact to a JSON sidecar."""
    meta = {
        k: v for k, v in bundle.items()
        if k not in {"up", "nonup"}
    }
    for head_name in ("up", "nonup"):
        head = bundle[head_name]
        meta[head_name] = {
            "feature_columns": head["feature_columns"],
            "feature_columns_hash": head["feature_columns_hash"],
            "best_iteration": head["best_iteration"],
            "validation_rho": head["validation_rho"],
            "n_train": head["n_train"],
            "n_val": head["n_val"],
        }
    out = sidecar_metadata_path(artifact_path)
    out.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out
