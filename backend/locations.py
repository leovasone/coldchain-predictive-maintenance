"""Fixed, synthetic fleet of cold-chain refrigeration assets spread across
several Latin American cities, plus a matching roster of field technicians.

Deliberately generic: no real client/brand name, no real street addresses.
City center coordinates are real (public knowledge), but every asset's exact
position is a small random offset around its city center, and every venue
name/neighborhood label is a generic, made-up placeholder -- there is no real
business or client behind any of this. This keeps the demo self-contained and
avoids implying a relationship with any real company.
"""
from __future__ import annotations

import random

# ── City centers (real public coordinates, used only as a scatter anchor) ────
CITIES = [
    {"city": "São Paulo", "country": "Brasil", "lat": -23.5505, "lon": -46.6333},
    {"city": "Cidade do México", "country": "México", "lat": 19.4326, "lon": -99.1332},
    {"city": "Bogotá", "country": "Colômbia", "lat": 4.7110, "lon": -74.0721},
    {"city": "Buenos Aires", "country": "Argentina", "lat": -34.6037, "lon": -58.3816},
    {"city": "Lima", "country": "Peru", "lat": -12.0464, "lon": -77.0428},
    {"city": "Santiago", "country": "Chile", "lat": -33.4489, "lon": -70.6693},
]

# ── Asset types this demo tracks, each with a distinct normal operating
# envelope and failure profile (used both by the telemetry simulator and the
# ML training data generator) ─────────────────────────────────────────────────
ASSET_TYPES = {
    "resfriador_bebidas": {
        "label": "Resfriador de Bebidas",
        "normal_temp_c": (2.0, 8.0),
        "normal_current_a": (1.5, 3.0),
        "door_events_per_day": (10, 60),
    },
    "freezer_sorvete": {
        "label": "Freezer de Sorvete",
        "normal_temp_c": (-22.0, -14.0),
        "normal_current_a": (2.5, 4.5),
        "door_events_per_day": (5, 30),
    },
    "camara_fria": {
        "label": "Câmara Fria",
        "normal_temp_c": (-2.0, 4.0),
        "normal_current_a": (5.0, 9.0),
        "door_events_per_day": (20, 90),
    },
}

ASSET_MODELS = {
    "resfriador_bebidas": ["ColdLine RB-300", "PolarStock RB-420", "FreshBev RB-250"],
    "freezer_sorvete": ["IceCore FZ-200", "GelatoMax FZ-310", "PolarStock FZ-280"],
    "camara_fria": ["ColdRoom CR-1000", "FreshHold CR-1500", "PolarStock CR-800"],
}

# Generic, made-up venue descriptors -- purely placeholder labels, not real
# businesses. Combined with a neutral zone label to build a display name.
VENUE_LABELS = [
    "Supermercado Regional", "Distribuidora de Bebidas", "Loja de Conveniência",
    "Mercado de Bairro", "Depósito Atacadista", "Restaurante Popular",
    "Padaria Central", "Sorveteria", "Açougue e Frios", "Minimercado",
]
ZONE_LABELS = ["Zona Norte", "Zona Sul", "Zona Leste", "Zona Oeste", "Centro", "Região Metropolitana"]

TECHNICIAN_SPECIALTIES = [
    "refrigeração e compressores",
    "sistemas elétricos",
    "eletrônica embarcada e sensores",
]

# First/last name pools kept generic and regionally neutral -- not modeled on
# any real roster.
_FIRST_NAMES = ["Marcos", "Luciana", "Rafael", "Camila", "Diego", "Valentina",
                "Andrés", "Sofía", "Bruno", "Patricia", "Emiliano", "Renata"]
_LAST_NAMES = ["Herrera", "Souza", "Vargas", "Oliveira", "Castillo", "Mendes",
               "Rojas", "Almeida", "Torres", "Figueroa", "Barros", "Salinas"]


def _scatter(lat: float, lon: float, rng: random.Random, spread_deg: float = 0.15) -> tuple[float, float]:
    """Small random offset around a city center -- enough spread to look like
    a real metro-area fleet without needing (or implying) real addresses."""
    return (
        round(lat + rng.uniform(-spread_deg, spread_deg), 4),
        round(lon + rng.uniform(-spread_deg, spread_deg), 4),
    )


def build_fleet(seed: int = 7, assets_per_city: int = 6) -> list[dict]:
    """Deterministic (seeded) synthetic fleet: assets_per_city assets in each
    of the 6 tracked cities, evenly split across the 3 asset types."""
    rng = random.Random(seed)
    fleet = []
    asset_type_cycle = list(ASSET_TYPES.keys())
    counter = 1
    for city in CITIES:
        for i in range(assets_per_city):
            asset_type = asset_type_cycle[i % len(asset_type_cycle)]
            lat, lon = _scatter(city["lat"], city["lon"], rng)
            venue = rng.choice(VENUE_LABELS)
            zone = rng.choice(ZONE_LABELS)
            fleet.append({
                "id": f"A-{counter:03d}",
                "name": f"{venue} — {zone}, {city['city']}",
                "city": city["city"],
                "country": city["country"],
                "lat": lat,
                "lon": lon,
                "asset_type": asset_type,
                "model": rng.choice(ASSET_MODELS[asset_type]),
                "install_year": rng.randint(2016, 2024),
                "status": "ok",
            })
            counter += 1
    return fleet


def build_technicians(seed: int = 11, per_city: int = 3) -> list[dict]:
    """Deterministic (seeded) synthetic technician roster, per_city
    technicians scattered near each tracked city."""
    rng = random.Random(seed)
    techs = []
    counter = 1
    for city in CITIES:
        for _ in range(per_city):
            lat, lon = _scatter(city["lat"], city["lon"], rng, spread_deg=0.2)
            name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
            techs.append({
                "id": counter,
                "name": name,
                "city": city["city"],
                "lat": lat,
                "lon": lon,
                "specialty": rng.choice(TECHNICIAN_SPECIALTIES),
                "status": "available",
            })
            counter += 1
    return techs


PARTS_CATALOG = [
    {"code": "CMP-100", "name": "Compressor selado 1/4 HP", "failure_types": ["temperatura_alta", "perda_refrigeracao"]},
    {"code": "GAS-100", "name": "Recarga de gás refrigerante R-290", "failure_types": ["temperatura_alta", "perda_refrigeracao"]},
    {"code": "TRM-100", "name": "Termostato digital", "failure_types": ["temperatura_alta", "temperatura_instavel"]},
    {"code": "VNT-100", "name": "Motor do ventilador do evaporador", "failure_types": ["temperatura_alta", "temperatura_instavel"]},
    {"code": "VED-100", "name": "Borracha de vedação da porta", "failure_types": ["temperatura_instavel", "excesso_aberturas"]},
    {"code": "ELT-100", "name": "Placa eletrônica de controle", "failure_types": ["falha_eletrica", "temperatura_instavel"]},
    {"code": "CAB-100", "name": "Cabo de alimentação blindado", "failure_types": ["falha_eletrica", "corrente_anomala"]},
    {"code": "CAP-100", "name": "Capacitor de partida", "failure_types": ["falha_eletrica", "corrente_anomala"]},
]


def parts_for_failure(failure_type: str) -> list[dict]:
    return [p for p in PARTS_CATALOG if failure_type in p.get("failure_types", [])]
