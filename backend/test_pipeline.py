"""Local smoke tests for the whole pipeline: synthetic fleet/technician
generation, the telemetry simulator, feature engineering, the two trained
ML models, both Claude agents (rule-based fallback path, since no API key
is available in this sandbox), and the FastAPI app's poll/alert/resolve
cycle end to end.

Everything here runs with no network access and no API key -- the same
"validate offline first" approach used by the sibling
realtime-weather-insights project, since this pipeline is also built to
degrade gracefully without either.

Run: python -m backend.test_pipeline
"""
from __future__ import annotations

import asyncio
import random

from . import diagnostics_agent, dispatch_agent, ml_models
from .features import extract_features
from .locations import build_fleet, build_technicians, parts_for_failure
from .telemetry_sim import AssetState, generate_training_sequences, next_reading


def test_fleet_and_technicians():
    """Fleet/technician generation must be deterministic (seeded) and
    reasonably shaped: every city represented, every asset type present,
    no duplicate IDs."""
    fleet = build_fleet()
    fleet_again = build_fleet()
    assert fleet == fleet_again, "build_fleet() must be deterministic for a given seed"

    ids = [a["id"] for a in fleet]
    assert len(ids) == len(set(ids)), "asset IDs must be unique"

    cities = {a["city"] for a in fleet}
    assert len(cities) == 6, f"expected 6 tracked cities, got {cities}"

    asset_types = {a["asset_type"] for a in fleet}
    assert asset_types == {"resfriador_bebidas", "freezer_sorvete", "camara_fria"}

    techs = build_technicians()
    tech_ids = [t["id"] for t in techs]
    assert len(tech_ids) == len(set(tech_ids)), "technician IDs must be unique"
    assert all(t["city"] in cities for t in techs), "every technician must be in a tracked city"

    parts = parts_for_failure("temperatura_alta")
    assert parts, "temperatura_alta must have at least one recommended part"
    print(f"[FLEET]  {len(fleet)} assets across {len(cities)} cities, {len(techs)} technicians")
    print("OK: fleet and technician generation is deterministic and well-shaped.\n")


def test_telemetry_normal_stays_in_envelope():
    """A non-degrading asset's readings should stay within (or very close
    to) its asset type's declared normal envelope -- if this drifted, the
    ML models would be trained on a "normal" that doesn't match what a
    healthy asset actually looks like at runtime."""
    from .locations import ASSET_TYPES

    rng = random.Random(1)
    state = AssetState(asset_id="A-TEST", asset_type="camara_fria")
    lo_t, hi_t = ASSET_TYPES["camara_fria"]["normal_temp_c"]

    for _ in range(50):
        reading = next_reading(state, rng, degrade_chance=0.0)
        assert lo_t - 0.5 <= reading.temperature_c <= hi_t + 0.5, (
            f"non-degrading reading {reading.temperature_c} fell outside the declared envelope "
            f"({lo_t}, {hi_t})"
        )
    print("OK: non-degrading telemetry stays within its asset type's declared envelope.\n")


def test_telemetry_degrading_drifts_out():
    """A forced-degrading asset should, by the end of its degradation
    window, be reading well outside its normal envelope -- otherwise the
    ML models have nothing real to learn to detect."""
    from .locations import ASSET_TYPES

    rng = random.Random(2)
    state = AssetState(asset_id="A-TEST2", asset_type="freezer_sorvete")
    state.degrading = True
    state.degrade_target_steps = 10
    _, hi_t = ASSET_TYPES["freezer_sorvete"]["normal_temp_c"]

    last_reading = None
    for _ in range(10):
        last_reading = next_reading(state, rng)
    assert last_reading.temperature_c > hi_t, (
        f"expected the end of a degradation window to read above the normal ceiling {hi_t}, "
        f"got {last_reading.temperature_c}"
    )
    print(f"[TELEMETRY]  degraded temperature={last_reading.temperature_c} (ceiling was {hi_t})")
    print("OK: forced degradation drifts telemetry well outside its normal envelope.\n")


def test_feature_extraction_shape():
    sequences, labels = generate_training_sequences("resfriador_bebidas", n_normal=3, n_failing=3, seq_len=6)
    feats = extract_features(sequences[0], "resfriador_bebidas")
    assert len(feats) == 9, f"expected 9 engineered features, got {len(feats)}"
    print("OK: feature extraction returns the expected fixed-length vector.\n")


def test_ml_models_available_and_separate_classes():
    """The two trained models must be loadable, and must actually separate
    a healthy window from a clearly-degrading one -- not just run without
    error."""
    assert ml_models.models_available(), "run `python -m backend.train_models` before this test"

    rng_normal = random.Random(10)
    state_normal = AssetState(asset_id="A-N", asset_type="camara_fria")
    normal_readings = [next_reading(state_normal, rng_normal, degrade_chance=0.0) for _ in range(8)]
    normal_score = ml_models.score(normal_readings, "camara_fria")

    rng_bad = random.Random(11)
    state_bad = AssetState(asset_id="A-B", asset_type="camara_fria")
    state_bad.degrading = True
    state_bad.degrade_target_steps = 8
    bad_readings = [next_reading(state_bad, rng_bad) for _ in range(8)]
    bad_score = ml_models.score(bad_readings, "camara_fria")

    assert bad_score["failure_probability"] > normal_score["failure_probability"], (
        "a degrading window must score a higher failure_probability than a healthy one"
    )
    assert "features" in bad_score and len(bad_score["features"]) == 9
    print(f"[ML]  normal failure_probability={normal_score['failure_probability']}  "
          f"degrading failure_probability={bad_score['failure_probability']}")
    print("OK: trained models load and correctly rank a degrading window above a healthy one.\n")


def test_diagnostics_fallback_varies_by_feature():
    """The rule-based fallback (used when ANTHROPIC_API_KEY isn't set, as
    in this sandbox) must classify differently for different dominant
    features, not collapse to the same alert_type every time -- this is
    the exact bug caught and fixed during development (see README)."""
    asset = {"name": "Test Asset", "city": "Lima", "country": "Peru",
              "asset_type": "resfriador_bebidas", "model": "TestModel", "install_year": 2020}

    high_temp = {"anomaly_score": 0.8, "failure_probability": 0.9,
                 "top_features": [], "features": {
                     "temp_dev_ratio": 2.5, "temp_trend_ratio": 0.1, "temp_std_ratio": 0.1,
                     "current_ratio": 0.9, "current_trend_ratio": 0.05,
                     "duty_last_pct": 60.0, "duty_trend_pct": 5.0,
                     "door_events_last": 1.0, "door_events_trend": 0.1}}
    high_current = {"anomaly_score": 0.8, "failure_probability": 0.9,
                    "top_features": [], "features": {
                        "temp_dev_ratio": 0.3, "temp_trend_ratio": 0.05, "temp_std_ratio": 0.05,
                        "current_ratio": 1.8, "current_trend_ratio": 0.7,
                        "duty_last_pct": 55.0, "duty_trend_pct": 3.0,
                        "door_events_last": 1.0, "door_events_trend": 0.1}}
    excess_doors = {"anomaly_score": 0.7, "failure_probability": 0.5,
                    "top_features": [], "features": {
                        "temp_dev_ratio": 0.2, "temp_trend_ratio": 0.05, "temp_std_ratio": 0.05,
                        "current_ratio": 0.9, "current_trend_ratio": 0.05,
                        "duty_last_pct": 50.0, "duty_trend_pct": 2.0,
                        "door_events_last": 6.0, "door_events_trend": 3.0}}

    diag_temp = diagnostics_agent.diagnose(asset, [], high_temp)
    diag_current = diagnostics_agent.diagnose(asset, [], high_current)
    diag_doors = diagnostics_agent.diagnose(asset, [], excess_doors)

    assert diag_temp["alert_type"] == "temperatura_alta", diag_temp
    assert diag_current["alert_type"] == "corrente_anomala", diag_current
    assert diag_doors["alert_type"] == "excesso_aberturas", diag_doors
    assert {diag_temp["alert_type"], diag_current["alert_type"], diag_doors["alert_type"]} == {
        "temperatura_alta", "corrente_anomala", "excesso_aberturas"
    }, "fallback classification must vary by which feature actually deviates, not collapse to one type"
    print(f"[DIAGNOSTICS]  temp->{diag_temp['alert_type']}  current->{diag_current['alert_type']}  "
          f"doors->{diag_doors['alert_type']}")
    print("OK: rule-based fallback classifies differently depending on which signal actually deviates.\n")


def test_dispatch_respects_region_radius():
    """A technician thousands of km away must never be dispatched --
    assets and technicians are scattered across separate Latin American
    cities, and a cross-country dispatch would be nonsensical. Also checks
    the specialty-priority rule and the graceful "no technician nearby"
    outcome."""
    asset = {"id": "A-1", "name": "Test Asset", "city": "Bogotá", "country": "Colômbia",
              "lat": 4.7110, "lon": -74.0721}

    near_wrong_specialty = {"id": 1, "name": "Tech A", "city": "Bogotá", "lat": 4.72, "lon": -74.08,
                             "specialty": "sistemas elétricos", "status": "available"}
    near_right_specialty = {"id": 2, "name": "Tech B", "city": "Bogotá", "lat": 4.75, "lon": -74.05,
                             "specialty": "refrigeração e compressores", "status": "available"}
    far_right_specialty = {"id": 3, "name": "Tech C", "city": "São Paulo", "lat": -23.55, "lon": -46.63,
                            "specialty": "refrigeração e compressores", "status": "available"}

    result = dispatch_agent.dispatch(
        "temperatura_alta", "high", "diagnóstico de teste", asset,
        [near_wrong_specialty, near_right_specialty, far_right_specialty],
    )
    assert result["technician_id"] == 2, (
        f"expected the nearby technician with matching specialty, got {result}"
    )
    assert result["distance_km"] < 150, "selected technician must be within the dispatch radius"

    only_far = dispatch_agent.dispatch(
        "temperatura_alta", "high", "diagnóstico de teste", asset, [far_right_specialty],
    )
    assert only_far["technician_id"] is None
    assert only_far["source"] == "no-technician-in-region"

    print(f"[DISPATCH]  selected={result['technician_name']} ({result['distance_km']}km), "
          f"far-only case correctly returned: {only_far['source']}")
    print("OK: dispatch prioritizes specialty+proximity and never crosses the region radius.\n")


def test_end_to_end_alert_lifecycle():
    """Drive the FastAPI app's own poll_once() enough times that at least
    one asset degrades, is diagnosed, dispatched, and (after the
    compressed simulated ETA) resolved -- exercising the exact code path
    the live dashboard runs, not a re-implementation of it."""
    from . import main as app_main

    async def run():
        app_main.SIM_SECONDS_PER_ETA_MINUTE = 0.01  # near-instant resolution for the test
        for _ in range(150):
            await app_main.poll_once()
        raised = app_main._alert_id_counter
        assert raised > 0, "expected at least one alert across 150 poll cycles for 36 assets"
        await asyncio.sleep(1.0)  # let pending _resolve_after tasks fire
        assert len(app_main.ACTIVE_ALERTS) < raised, "at least one alert should have resolved by now"
        avail = sum(1 for t in app_main.TECHNICIANS.values() if t["status"] == "available")
        assert avail == len(app_main.TECHNICIANS), "every technician should be free again once all alerts resolve"
        return raised, len(app_main.ACTIVE_ALERTS)

    raised, still_active = asyncio.run(run())
    print(f"[END-TO-END]  alerts raised={raised}  still active after settling={still_active}")
    print("OK: poll -> diagnose -> dispatch -> resolve runs end to end through the real app module.\n")


def main():
    test_fleet_and_technicians()
    test_telemetry_normal_stays_in_envelope()
    test_telemetry_degrading_drifts_out()
    test_feature_extraction_shape()
    test_ml_models_available_and_separate_classes()
    test_diagnostics_fallback_varies_by_feature()
    test_dispatch_respects_region_radius()
    test_end_to_end_alert_lifecycle()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
