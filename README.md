# Cold-Chain Predictive Maintenance

A working demo of predictive maintenance for refrigeration assets (beverage
coolers, ice cream freezers, cold rooms) spread across several Latin
American cities: synthetic IoT telemetry, two trained Machine Learning
models (unsupervised anomaly detection + supervised failure-risk
prediction), and two Claude-based agents (diagnosis, then field-technician
dispatch), all streamed live to a browser dashboard over WebSockets.

Everything here is synthetic end to end. There is no real fleet, no real
IoT hardware, no real technicians, and no real client or brand behind this
-- it exists purely to demonstrate the architecture pattern (IoT ingestion
-> ML scoring -> agentic decision-making -> live dashboard) with data
generated specifically for this project.

## Why Latin America, distributed

Rather than inventing one fictitious city, the fleet is spread across six
real Latin American metro areas (São Paulo, Cidade do México, Bogotá,
Buenos Aires, Lima, Santiago). Every asset's exact position, venue name,
and "neighborhood" label is a randomized placeholder -- the city centers
are the only real-world detail, used purely as a scatter anchor.

## Architecture

```
telemetry_sim.py (synthetic IoT)  ->  features.py  ->  ml_models.py
                                                          |  (anomaly_score, failure_probability)
                                                          v
                                              diagnostics_agent.py (ColdSentinel)
                                                          |  (alert_type, severity, diagnosis)
                                                          v
                                                dispatch_agent.py (FieldPilot)
                                                          |  (technician, ETA, parts)
                                                          v
                                          WebSocket broadcast -> browser dashboard
```

- **`backend/locations.py`** -- deterministic (seeded) synthetic fleet of
  ~36 assets and 18 technicians distributed across the 6 tracked cities,
  plus a parts catalog. No real addresses or business names.
- **`backend/telemetry_sim.py`** -- generates per-asset temperature,
  current draw, compressor duty cycle, and door-open-rate readings. Most
  assets stay healthy; a small fraction enter a slow, multi-cycle
  "degrading" state each poll, drifting toward failure before "getting
  repaired" (reset to healthy). Also generates the labeled sequences used
  to train the two ML models.
- **`backend/features.py`** -- shared feature engineering (used by both
  training and runtime scoring) that expresses every raw signal *relative
  to* its asset type's own normal operating envelope, so one pair of
  models can be trained across all three very different asset types at
  once.
- **`backend/train_models.py`** -- offline training script: an
  `IsolationForest` (genuinely unsupervised, trained only on normal
  sequences) and a `GradientBoostingClassifier` (genuinely supervised,
  trained on labeled normal/failing sequences), both on synthetic data.
  Produces the committed `backend/models/*.joblib` artifacts plus
  `metadata.json` (feature importances, held-out ROC-AUC).
- **`backend/ml_models.py`** -- loads the trained artifacts and scores a
  live window of readings: `anomaly_score`, `failure_probability`, and
  both the model's global top features and this specific instance's own
  feature values.
- **`backend/diagnostics_agent.py`** -- **ColdSentinel**: takes the ML
  output and turns it into a plain-language diagnosis and an alert
  category. Falls back to an explicit rule-based classification (driven
  by which of *this instance's* features actually deviates, not just the
  model's fixed global importance ranking) if no `ANTHROPIC_API_KEY` is
  set or the API call fails.
- **`backend/dispatch_agent.py`** -- **FieldPilot**: picks the best
  available technician (specialty match, then proximity via plain
  Haversine in `geo.py`) and recommends parts. Candidates outside a
  150km radius are never considered, even as a fallback -- assets and
  technicians are spread across different countries, and a cross-country
  dispatch would be nonsensical.
- **`backend/main.py`** -- FastAPI app: a background loop advances every
  asset's simulated telemetry, scores it, and escalates through both
  agents when failure_probability crosses a threshold. A new WebSocket
  client gets the full current fleet/technician/alert snapshot
  immediately on connect (not just future updates).
- **`frontend/index.html`** -- single-page dashboard: a Latin America map
  (Leaflet, no API key) with live asset/technician markers, fleet
  status counts, a live alert/dispatch feed, and an "ML insights" panel
  showing the trained models' held-out accuracy and feature importances.

## Running locally

```bash
pip install -r backend/requirements.txt
python -m backend.train_models   # only needed once, or after changing telemetry_sim.py
uvicorn backend.main:app --reload
```

Then open `http://localhost:8000`. Set `ANTHROPIC_API_KEY` to enable the
two Claude agents; without it, both fall back to their rule-based logic
and the dashboard runs identically otherwise.

## Testing without live network access

```bash
python -m backend.test_pipeline
```

Covers fleet/technician generation, the telemetry simulator's normal vs.
degrading envelopes, feature extraction, both trained ML models actually
separating a healthy window from a degrading one, the diagnostics
fallback's per-instance classification, the dispatch agent's
specialty+radius logic, and a full poll -> alert -> dispatch -> resolve
cycle run through the real `main.py` module (not a re-implementation of
it).

## Deploying to Railway

1. Push this repo to GitHub.
2. In Railway, **New Project -> Deploy from GitHub repo**. Railway
   auto-detects the `Dockerfile`.
3. `ANTHROPIC_API_KEY` is optional (enables the two Claude agents; add it
   in Railway's Variables tab). Without it, the dashboard runs the same,
   just with rule-based diagnosis/dispatch instead of Claude's.
4. `backend/models/*.joblib` are committed to the repo, so no training
   step is needed at deploy time.
5. Fleet/technician/alert state lives in memory and resets on every
   restart -- fine for a demo, same tradeoff documented in this
   portfolio's other live-dashboard project.

## Honest notes

- Nothing here is real: no real fleet, no real IoT sensors, no real
  technicians, no real client or brand. The city center coordinates are
  the only real-world data point, used only to scatter synthetic assets
  around six real metro areas.
- The held-out ROC-AUC for the failure-risk classifier comes back at
  1.000 on this synthetic dataset -- that reflects how clean and
  deterministic the synthetic degradation curve is, not a claim about how
  well this approach would generalize to a real fleet's noisier history.
  A real deployment would need real historical failure data and would
  almost certainly see a meaningfully lower score.
- `anomaly_score` is `IsolationForest.decision_function()` rescaled into a
  0-1 range for display convenience. It is **not** a calibrated
  probability the way `failure_probability` (from the supervised
  classifier's `predict_proba`) is -- the two numbers are not directly
  comparable.
- The two ML models are trained once across all three asset types at
  once, using features expressed relative to each asset type's own normal
  envelope (see `features.py`) rather than raw physical units -- this
  avoids needing three separate models while keeping "1.5x above the
  normal ceiling" meaningful regardless of whether the asset normally
  runs at 5°C or -18°C.
- The diagnostics fallback originally picked `alert_type` from the ML
  model's *global* top-3 feature importances, which are the same three
  features every time regardless of the instance -- so every fallback
  diagnosis collapsed to the same category ("temperatura_alta") no matter
  what was actually anomalous about a given asset. Fixed by exposing each
  instance's own full feature vector and running an explicit threshold
  cascade over *those* values instead, which `test_pipeline.py` now pins
  directly (three different synthetic feature profiles must yield three
  different alert types).
- The dispatch agent initially picked the nearest available technician
  from the *entire* fleet-wide roster with no distance cap. Since assets
  and technicians are spread across six different countries, this could
  (and did, in testing) "dispatch" a technician over 1,600km away. Fixed
  by capping candidates to a 150km radius and returning an explicit
  "no technician in region" result instead of a cross-country pick when
  none qualify.
- Dispatch ETAs (10-100+ real minutes) are compressed into a handful of
  demo-seconds (`SIM_SECONDS_PER_ETA_MINUTE`) purely so a live viewer
  doesn't have to wait real minutes to see an alert resolve. This is a
  UI/demo pacing choice, not a claim about real repair times.
- Following the same fix already made in this portfolio's
  realtime-weather-insights project: a new WebSocket client gets the full
  current fleet/technician/active-alert snapshot immediately on connect,
  rather than only receiving future updates and potentially staring at an
  empty dashboard until the next poll cycle.
