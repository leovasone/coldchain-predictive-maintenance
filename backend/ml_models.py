"""Runtime scoring using the two models trained by `train_models.py`.

Loaded once at import time from the committed `models/*.joblib` artifacts.
If the artifacts are missing (e.g. a fresh checkout before running the
training script), this degrades to returning `None` scores rather than
crashing the app -- same "degrade, don't break" pattern used throughout
this portfolio's other projects.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib

from .features import FEATURE_NAMES, extract_features
from .telemetry_sim import Reading

log = logging.getLogger("ml_models")

MODELS_DIR = Path(__file__).resolve().parent / "models"

_anomaly_model = None
_failure_model = None
_metadata: dict | None = None


def _load():
    global _anomaly_model, _failure_model, _metadata
    try:
        _anomaly_model = joblib.load(MODELS_DIR / "anomaly_model.joblib")
        _failure_model = joblib.load(MODELS_DIR / "failure_model.joblib")
        _metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
        log.info("ml_models: loaded trained models (held_out_roc_auc=%s)", _metadata.get("held_out_roc_auc"))
    except FileNotFoundError:
        log.warning("ml_models: no trained model artifacts found -- run `python -m backend.train_models` first")


_load()


def models_available() -> bool:
    return _anomaly_model is not None and _failure_model is not None


def metadata() -> dict | None:
    return _metadata


def score(readings: list[Reading], asset_type: str) -> dict | None:
    """Score a window of readings (oldest first) for one asset.

    Returns a dict with:
    - anomaly_score: 0-1, higher = further from the shape of normal
      operation the IsolationForest learned (unsupervised).
    - failure_probability: 0-1, the GradientBoostingClassifier's
      predict_proba for "this window ends in a failure state" (supervised).
    - top_features: the 3 globally most important features for the
      failure-risk model, alongside this window's own value for each --
      global importance, not a per-prediction explanation (SHAP-style),
      and the README says so explicitly.

    Returns None if the models haven't been loaded (see `_load` above).
    """
    if not models_available():
        return None

    feats = extract_features(readings, asset_type)
    X = [feats]

    # IsolationForest's decision_function is higher = more normal; flip and
    # squash into a 0-1 "anomaly score" that's easier for the frontend/agent
    # to reason about (0 = looks normal, 1 = very anomalous relative to the
    # training distribution). This rescaling is a display convenience, not
    # a probability -- the README is explicit that this is not a calibrated
    # probability the way failure_probability is.
    raw = _anomaly_model.decision_function(X)[0]
    anomaly_score = max(0.0, min(1.0, 0.5 - raw))

    failure_probability = float(_failure_model.predict_proba(X)[0][1])

    top_features = []
    for item in (_metadata or {}).get("feature_importances", [])[:3]:
        name = item["feature"]
        idx = FEATURE_NAMES.index(name)
        top_features.append({
            "feature": name,
            "importance": item["importance"],
            "value": round(feats[idx], 3),
        })

    return {
        "anomaly_score": round(anomaly_score, 3),
        "failure_probability": round(failure_probability, 3),
        "top_features": top_features,
        # Full feature vector for this instance, keyed by name -- used by
        # diagnostics_agent's rule-based fallback (and the LLM prompt) to
        # reason about *which* signal is driving this specific alert, not
        # just the model's fixed global importance ranking (which is the
        # same 3 features every time and doesn't vary per instance).
        "features": dict(zip(FEATURE_NAMES, [round(v, 3) for v in feats])),
    }
