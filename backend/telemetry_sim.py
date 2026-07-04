"""Synthetic IoT telemetry generator for cold-chain refrigeration assets.

Two jobs, both built on the same underlying model:

1. Live demo simulation (`next_reading`): each poll cycle, most assets stay
   in their normal operating envelope; a small number are put into a slow
   "degrading" state (temperature drifting up, current drawing more, door
   events climbing) so the dashboard has something real for the ML models
   and agents to catch, instead of every asset being permanently healthy.

2. Offline training data (`generate_training_sequences`): produces many
   synthetic run sequences -- most entirely normal, some ending in a
   labeled failure -- used only to train the two models in
   `train_models.py`. Never touches a real fleet's history (there is no
   real fleet here); it exists purely so the ML component is trained on
   *something* resembling degradation physics instead of pure noise.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .locations import ASSET_TYPES


@dataclass
class AssetState:
    """Per-asset simulation state carried between poll cycles -- tracks
    whether an asset is currently degrading and how far along that is, so
    successive readings drift smoothly instead of jumping randomly."""
    asset_id: str
    asset_type: str
    degrading: bool = False
    degrade_step: int = 0
    degrade_target_steps: int = 0
    door_events_today: int = 0


@dataclass
class Reading:
    asset_id: str
    temperature_c: float
    current_a: float
    door_events_last_hour: int
    compressor_duty_cycle_pct: float


def _normal_reading(asset_type: str, rng: random.Random) -> Reading:
    spec = ASSET_TYPES[asset_type]
    lo_t, hi_t = spec["normal_temp_c"]
    lo_c, hi_c = spec["normal_current_a"]
    lo_d, hi_d = spec["door_events_per_day"]
    return Reading(
        asset_id="",
        temperature_c=round(rng.uniform(lo_t, hi_t), 1),
        current_a=round(rng.uniform(lo_c, hi_c), 2),
        door_events_last_hour=round(rng.uniform(lo_d, hi_d) / 24, 1),
        compressor_duty_cycle_pct=round(rng.uniform(30, 55), 1),
    )


def _degrading_reading(asset_type: str, progress: float, rng: random.Random) -> Reading:
    """`progress` in [0, 1]: 0 = just started degrading, 1 = about to fail.
    Temperature and current drift upward past the normal envelope, duty
    cycle climbs toward 100% (compressor running near-constantly trying to
    compensate), and door events get noisier -- a plausible (not
    medically/thermodynamically rigorous) stand-in for a compressor or
    seal slowly losing efficiency."""
    spec = ASSET_TYPES[asset_type]
    lo_t, hi_t = spec["normal_temp_c"]
    lo_c, hi_c = spec["normal_current_a"]
    span_t = hi_t - lo_t
    drift_t = span_t * (0.4 + 1.6 * progress)  # can push well past hi_t
    temp = hi_t - span_t * 0.3 + drift_t * rng.uniform(0.7, 1.1)
    current = hi_c * (1.0 + 0.6 * progress) * rng.uniform(0.9, 1.15)
    duty = min(100.0, 55 + 45 * progress * rng.uniform(0.85, 1.1))
    door = rng.uniform(0.5, 4.0) * (1 + progress)
    return Reading(
        asset_id="",
        temperature_c=round(temp, 1),
        current_a=round(current, 2),
        door_events_last_hour=round(door, 1),
        compressor_duty_cycle_pct=round(duty, 1),
    )


def next_reading(state: AssetState, rng: random.Random, degrade_chance: float = 0.01,
                  degrade_len_range: tuple[int, int] = (10, 20)) -> Reading:
    """Advance one asset's simulation by one poll cycle and return its new
    reading. A healthy asset has `degrade_chance` probability of starting a
    slow degradation each cycle; once degrading, it walks toward failure
    over `degrade_len_range` steps before resetting (simulating a repair)."""
    if not state.degrading:
        if rng.random() < degrade_chance:
            state.degrading = True
            state.degrade_step = 0
            state.degrade_target_steps = rng.randint(*degrade_len_range)
        reading = _normal_reading(state.asset_type, rng)
    else:
        progress = min(1.0, state.degrade_step / max(1, state.degrade_target_steps))
        reading = _degrading_reading(state.asset_type, progress, rng)
        state.degrade_step += 1
        if state.degrade_step > state.degrade_target_steps:
            # "Repaired": back to normal for the next cycle.
            state.degrading = False
            state.degrade_step = 0

    reading.asset_id = state.asset_id
    return reading


def generate_training_sequences(asset_type: str, n_normal: int = 400, n_failing: int = 150,
                                 seq_len: int = 12, seed: int = 3) -> tuple[list[list[Reading]], list[int]]:
    """Build labeled sequences purely for offline model training:
    `n_normal` sequences of `seq_len` healthy readings (label 0), and
    `n_failing` sequences that walk from healthy toward failure over
    `seq_len` steps (label 1 -- "this sequence ends in a failure state").
    Synthetic by construction; see train_models.py and the README for why
    this is labeled as training on synthetic data, not real fleet history.
    """
    rng = random.Random(seed)
    sequences: list[list[Reading]] = []
    labels: list[int] = []

    for _ in range(n_normal):
        seq = [_normal_reading(asset_type, rng) for _ in range(seq_len)]
        sequences.append(seq)
        labels.append(0)

    for _ in range(n_failing):
        seq = []
        for step in range(seq_len):
            progress = step / (seq_len - 1)
            seq.append(_degrading_reading(asset_type, progress, rng))
        sequences.append(seq)
        labels.append(1)

    return sequences, labels
