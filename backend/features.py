"""Shared feature engineering for the two ML models (`train_models.py` and
`ml_models.py` both import this -- training and serving must compute
features identically, or the trained model silently sees different inputs
at inference time than it did during training).

Every raw signal (temperature, current, duty cycle, door events) is
expressed *relative to* the asset type's own normal operating envelope
(see `locations.ASSET_TYPES`) rather than as a raw physical unit. That is
what lets a single pair of models be trained across all three asset types
(beverage cooler, ice cream freezer, cold room) at once, despite their very
different normal temperature/current ranges -- "1.5x above the normal
ceiling" means the same thing to the model whether the asset normally runs
at 5°C or -18°C.
"""
from __future__ import annotations

from .locations import ASSET_TYPES
from .telemetry_sim import Reading

FEATURE_NAMES = [
    "temp_dev_ratio",       # last reading's distance from the envelope's center, in half-widths
    "temp_trend_ratio",     # how much temperature moved over the window, in half-widths
    "temp_std_ratio",       # temperature volatility over the window, in half-widths
    "current_ratio",        # last current draw as a fraction of the envelope's normal ceiling
    "current_trend_ratio",  # how much current draw moved over the window
    "duty_last_pct",        # last compressor duty cycle (0-100)
    "duty_trend_pct",       # how much duty cycle moved over the window
    "door_events_last",     # last door-open rate (events/hour)
    "door_events_trend",    # how much the door-open rate moved over the window
]


def _envelope(asset_type: str) -> tuple[float, float, float]:
    spec = ASSET_TYPES[asset_type]
    lo_t, hi_t = spec["normal_temp_c"]
    _, hi_c = spec["normal_current_a"]
    center_t = (lo_t + hi_t) / 2
    half_t = (hi_t - lo_t) / 2
    return center_t, half_t, hi_c


def extract_features(readings: list[Reading], asset_type: str) -> list[float]:
    """`readings` must be in chronological order, oldest first, length >= 2."""
    if len(readings) < 2:
        raise ValueError("extract_features needs at least 2 readings to compute a trend")

    center_t, half_t, hi_c = _envelope(asset_type)
    temps = [r.temperature_c for r in readings]
    currents = [r.current_a for r in readings]
    duties = [r.compressor_duty_cycle_pct for r in readings]
    doors = [r.door_events_last_hour for r in readings]

    mean_t = sum(temps) / len(temps)
    var_t = sum((t - mean_t) ** 2 for t in temps) / len(temps)

    return [
        (temps[-1] - center_t) / half_t,
        (temps[-1] - temps[0]) / half_t,
        (var_t ** 0.5) / half_t,
        currents[-1] / hi_c,
        (currents[-1] - currents[0]) / hi_c,
        duties[-1],
        duties[-1] - duties[0],
        doors[-1],
        doors[-1] - doors[0],
    ]
