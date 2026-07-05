"""FieldPilot — the technician dispatch agent.

Given a diagnosed alert (from `diagnostics_agent.py`) and the current
roster of technicians, picks who should respond and which parts they
should bring. Distance is computed with plain Haversine (`geo.py`); the
LLM's job is to weigh distance against specialty match and urgency, not to
estimate geography itself.

Entirely optional, same "degrade, don't break" pattern as
`diagnostics_agent.py`: no API key or a failed call falls back to a
rule-based "nearest technician with a matching specialty, or nearest
overall" selection.
"""
from __future__ import annotations

import json
import logging
import os
import re

import anthropic

from .geo import haversine_km
from .locations import parts_for_failure

log = logging.getLogger("dispatch_agent")

# Assets are spread across several Latin American cities, thousands of km
# apart -- a technician only makes sense as a dispatch candidate within their
# own metro area, never across countries. Candidates beyond this radius are
# filtered out before either the rule-based fallback or the LLM ever sees
# them, rather than trusting the model to reject a nonsensical cross-country
# suggestion on its own.
MAX_DISPATCH_RADIUS_KM = 150.0

_client: anthropic.Anthropic | None = None
if os.environ.get("ANTHROPIC_API_KEY"):
    _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

_SYSTEM_PROMPT = """Você é o FieldPilot, agente de envio de técnicos de campo para
uma frota de ativos de refrigeração monitorados por IoT na América Latina.

Sua função: dado um alerta já diagnosticado e uma lista de técnicos disponíveis
(com distância já calculada), escolher o melhor técnico e as peças necessárias.

## Critérios de seleção:
1. Priorizar especialidade compatível com o tipo de falha
   (ex.: "refrigeração e compressores" para temperatura_alta/perda_refrigeracao;
   "sistemas elétricos" ou "eletrônica embarcada e sensores" para falha_eletrica/corrente_anomala)
2. Entre técnicos com especialidade adequada, priorizar o mais próximo
3. Se nenhum técnico tiver especialidade ideal, escolher o mais próximo disponível

## Resposta em JSON:
{
  "selected_technician_id": <int>,
  "reasoning": "2-3 frases: por que este técnico, urgência, risco ao produto se não atendido a tempo",
  "parts": [{"code": "XXX-000", "name": "nome da peça", "qty": 1, "reason": "por que essa peça"}],
  "priority_note": "mensagem curta para o técnico sobre urgência e cuidados"
}

Retorne APENAS JSON válido, sem texto adicional."""


def dispatch(alert_type: str, severity: str, diagnosis: str, asset: dict,
             available_technicians: list[dict]) -> dict:
    """Returns technician_id/technician_name/distance_km/eta_min/parts/reasoning."""
    if not available_technicians:
        return _no_technician_available()

    techs_with_dist = sorted(
        (
            {**t, "distance_km": round(haversine_km(asset["lat"], asset["lon"], t["lat"], t["lon"]), 1)}
            for t in available_technicians
        ),
        key=lambda t: t["distance_km"],
    )
    techs_with_dist = [t for t in techs_with_dist if t["distance_km"] <= MAX_DISPATCH_RADIUS_KM]
    if not techs_with_dist:
        return _no_technician_in_region()

    parts_catalog = parts_for_failure(alert_type)

    if _client is None:
        return _fallback(techs_with_dist, parts_catalog, asset, alert_type)

    tech_list = "\n".join(
        f"  ID {t['id']}: {t['name']} — {t['specialty']} — {t['distance_km']}km — {t['city']}"
        for t in techs_with_dist[:6]
    )
    parts_list = "\n".join(f"  {p['code']}: {p['name']}" for p in parts_catalog) or "  Kit de manutenção geral"

    user_msg = f"""Alerta:
- Ativo: {asset['name']} ({asset['city']}, {asset['country']})
- Tipo de falha: {alert_type}
- Severidade: {severity}
- Diagnóstico: {diagnosis}

Técnicos disponíveis (ordenados por distância):
{tech_list}

Peças candidatas para este tipo de falha:
{parts_list}

Selecione o técnico ideal e as peças necessárias."""

    try:
        response = _client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text.strip()
        if "```" in raw:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
            if m:
                raw = m.group(1).strip()
        data = json.loads(raw)

        selected_id = int(data.get("selected_technician_id", techs_with_dist[0]["id"]))
        tech = next((t for t in techs_with_dist if t["id"] == selected_id), techs_with_dist[0])
        eta_min = max(10, int(tech["distance_km"] / 35 * 60 + 8))  # 35 km/h avg + 8min prep

        parts = data.get("parts") or [
            {"code": p["code"], "name": p["name"], "qty": 1, "reason": "recomendado para este tipo de falha"}
            for p in parts_catalog[:3]
        ]

        return {
            "technician_id": tech["id"],
            "technician_name": tech["name"],
            "distance_km": tech["distance_km"],
            "eta_min": eta_min,
            "parts": parts,
            "reasoning": str(data.get("reasoning", "")),
            "priority_note": str(data.get("priority_note", "")),
            "source": "claude-haiku-4-5",
        }
    except Exception as exc:
        log.warning("dispatch_agent: Claude call failed, using rule-based fallback: %s", exc)
        return _fallback(techs_with_dist, parts_catalog, asset, alert_type)


_SPECIALTY_FOR_ALERT = {
    "temperatura_alta": "refrigeração e compressores",
    "perda_refrigeracao": "refrigeração e compressores",
    "temperatura_instavel": "refrigeração e compressores",
    "excesso_aberturas": "refrigeração e compressores",
    "falha_eletrica": "sistemas elétricos",
    "corrente_anomala": "eletrônica embarcada e sensores",
}


def _fallback(techs_with_dist: list[dict], parts_catalog: list[dict], asset: dict, alert_type: str) -> dict:
    wanted_specialty = _SPECIALTY_FOR_ALERT.get(alert_type)
    matching = [t for t in techs_with_dist if t["specialty"] == wanted_specialty]
    tech = matching[0] if matching else techs_with_dist[0]
    eta_min = max(10, int(tech["distance_km"] / 35 * 60 + 8))
    parts = [{"code": p["code"], "name": p["name"], "qty": 1, "reason": "recomendado"} for p in parts_catalog[:2]]
    return {
        "technician_id": tech["id"],
        "technician_name": tech["name"],
        "distance_km": tech["distance_km"],
        "eta_min": eta_min,
        "parts": parts,
        "reasoning": (
            f"Técnico com especialidade compatível mais próximo selecionado ({tech['distance_km']}km). "
            f"Atendimento para {asset['name']}."
        ),
        "priority_note": "Selecionado por regra fixa (agente de IA indisponível).",
        "source": "rule-based-fallback",
    }


def _no_technician_available() -> dict:
    return {
        "technician_id": None,
        "technician_name": None,
        "distance_km": None,
        "eta_min": None,
        "parts": [],
        "reasoning": "Nenhum técnico disponível no momento.",
        "priority_note": "Alerta em fila até liberação de um técnico.",
        "source": "no-technician-available",
    }


def _no_technician_in_region() -> dict:
    return {
        "technician_id": None,
        "technician_name": None,
        "distance_km": None,
        "eta_min": None,
        "parts": [],
        "reasoning": (
            f"Nenhum técnico disponível dentro de {MAX_DISPATCH_RADIUS_KM:.0f}km deste ativo "
            "no momento -- todos os técnicos da região já estão em atendimento."
        ),
        "priority_note": "Alerta em fila até liberação de um técnico na região.",
        "source": "no-technician-in-region",
    }
