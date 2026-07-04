"""ColdSentinel — the diagnosis agent.

Takes the ML layer's output (anomaly score, failure probability, top
contributing features -- see `ml_models.py`) plus the asset's raw recent
readings, and turns them into a plain-language technical diagnosis and an
alert classification. The ML models decide *how anomalous* and *how
likely to fail*; this agent's only job is to explain that in words and
pick a category/severity a technician can act on -- it never invents its
own anomaly judgment independent of the ML scores it's given.

Entirely optional: with no `ANTHROPIC_API_KEY` set, or if the API call
fails, `diagnose()` falls back to a rule-based classification derived
directly from the same ML output -- same "degrade, don't break" pattern
used by every other AI-optional feature in this portfolio.
"""
from __future__ import annotations

import json
import logging
import os
import re

import anthropic

log = logging.getLogger("diagnostics_agent")

_client: anthropic.Anthropic | None = None
if os.environ.get("ANTHROPIC_API_KEY"):
    _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

_SYSTEM_PROMPT = """Você é o ColdSentinel, um agente de diagnóstico técnico para uma
frota de ativos de refrigeração (resfriadores de bebida, freezers de sorvete,
câmaras frias) monitorados por IoT.

Você recebe o resultado de dois modelos de Machine Learning já treinados
(não invente seu próprio julgamento de anomalia):
- anomaly_score (0-1): quão fora do padrão normal esse ativo está, segundo um
  modelo não-supervisionado (IsolationForest).
- failure_probability (0-1): probabilidade estimada de falha, segundo um
  modelo supervisionado (Gradient Boosting) treinado em dados sintéticos.
- top_features: as features mais importantes para o modelo, com o valor atual
  de cada uma neste ativo.

Sua função: traduzir esses números em um diagnóstico técnico claro e
categorizar o alerta.

## Categorias de alert_type (escolha uma):
temperatura_alta, temperatura_instavel, perda_refrigeracao, excesso_aberturas,
falha_eletrica, corrente_anomala

## Severidade:
- medium: failure_probability < 0.4
- high: failure_probability entre 0.4 e 0.75
- critical: failure_probability > 0.75

## Resposta em JSON:
{
  "alert_type": "uma das categorias acima",
  "severity": "medium|high|critical",
  "diagnosis": "diagnóstico técnico em 2-3 frases, citando os números que mais pesaram",
  "recommended_action": "ação imediata recomendada"
}

Retorne APENAS JSON válido, sem texto adicional."""


def diagnose(asset: dict, readings: list, ml_result: dict) -> dict:
    """`asset` is a fleet entry (see locations.build_fleet), `readings` is
    the recent window of Reading objects, `ml_result` is ml_models.score()'s
    output. Returns alert_type/severity/diagnosis/recommended_action."""
    if _client is None:
        return _fallback(ml_result)

    temp_series = ", ".join(f"{r.temperature_c:.1f}°C" for r in readings[-5:])
    current_series = ", ".join(f"{r.current_a:.2f}A" for r in readings[-5:])
    all_features = ml_result.get("features", {})
    features_str = "\n".join(f"  {name}: {value}" for name, value in all_features.items())
    top_str = "\n".join(
        f"  {f['feature']} (importância global do modelo: {f['importance']})"
        for f in ml_result.get("top_features", [])
    )

    user_msg = f"""Ativo: {asset['name']} ({asset['city']}, {asset['country']})
Tipo: {asset['asset_type']} — modelo {asset['model']}, instalado em {asset['install_year']}

Resultado dos modelos de ML:
- anomaly_score: {ml_result['anomaly_score']}
- failure_probability: {ml_result['failure_probability']}
- todas as features deste ativo agora:
{features_str}
- features mais importantes para o modelo em geral (não necessariamente as que mais pesam neste caso):
{top_str}

Últimas leituras (temperatura): {temp_series}
Últimas leituras (corrente): {current_series}

Classifique o alerta com base em qual(is) feature(s) estão mais fora do normal
*neste ativo específico*, e forneça o diagnóstico técnico."""

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
        return {
            "alert_type": str(data.get("alert_type", "temperatura_alta")),
            "severity": str(data.get("severity", _severity_from_probability(ml_result["failure_probability"]))),
            "diagnosis": str(data.get("diagnosis", "Anomalia detectada pelo modelo de ML.")),
            "recommended_action": str(data.get("recommended_action", "Inspecionar o ativo.")),
            "source": "claude-haiku-4-5",
        }
    except Exception as exc:
        log.warning("diagnostics_agent: Claude call failed, using rule-based fallback: %s", exc)
        return _fallback(ml_result)


def _severity_from_probability(p: float) -> str:
    if p > 0.75:
        return "critical"
    if p > 0.4:
        return "high"
    return "medium"


def _classify_from_features(features: dict) -> tuple[str, str]:
    """Explicit threshold cascade over *this instance's own* feature values
    (not the model's fixed global importance ranking, which is the same 3
    features every time and would otherwise make every fallback diagnosis
    land on the same alert_type regardless of what's actually unusual about
    this particular asset). Returns (alert_type, a short reason written in
    plain business language -- this string is shown directly to viewers of
    the dashboard, so it deliberately avoids raw variable names/code
    syntax like `temp_dev_ratio=1.4`)."""
    temp_dev = features.get("temp_dev_ratio", 0.0)
    temp_trend = features.get("temp_trend_ratio", 0.0)
    temp_std = features.get("temp_std_ratio", 0.0)
    current_ratio = features.get("current_ratio", 0.0)
    current_trend = features.get("current_trend_ratio", 0.0)
    duty_last = features.get("duty_last_pct", 0.0)
    door_last = features.get("door_events_last", 0.0)
    door_trend = features.get("door_events_trend", 0.0)

    if temp_dev > 1.2 and current_ratio > 1.3:
        return "falha_eletrica", (
            f"desvio de temperatura acentuado ({temp_dev:.1f}x o normal) combinado com "
            f"corrente elétrica {current_ratio:.1f}x acima do limite -- padrão típico de falha elétrica"
        )
    if current_ratio > 1.4 or abs(current_trend) > 0.5:
        return "corrente_anomala", (
            f"corrente elétrica em {current_ratio:.1f}x o limite normal "
            f"(variação de {current_trend:+.1f} na janela recente)"
        )
    if temp_std > 0.5 or abs(temp_trend) > 0.9:
        return "temperatura_instavel", (
            f"temperatura instável -- variação de {temp_std:.1f} e tendência de {temp_trend:+.1f} "
            "acima do esperado para operação normal"
        )
    if door_last > 3.0 or door_trend > 2.0:
        return "excesso_aberturas", (
            f"{door_last:.0f} aberturas de porta na última hora (tendência de {door_trend:+.1f}) -- "
            "acima do padrão de uso normal"
        )
    if duty_last > 90.0:
        return "perda_refrigeracao", f"compressor operando em {duty_last:.0f}% do ciclo, próximo do limite"
    if temp_dev > 0.8:
        return "temperatura_alta", f"temperatura {temp_dev:.1f}x acima do centro da faixa normal"
    return "temperatura_alta", (
        f"leve desvio de temperatura ({temp_dev:.1f}x o normal) -- sinal mais relevante "
        "disponível, sem indicador mais específico no momento"
    )


def _fallback(ml_result: dict) -> dict:
    """Rule-based classification derived from the same ML scores, used when
    no API key is configured or the Claude call fails."""
    severity = _severity_from_probability(ml_result["failure_probability"])
    features = ml_result.get("features", {})
    alert_type, reason = _classify_from_features(features)
    anomaly_pct = round(ml_result["anomaly_score"] * 100)
    failure_pct = round(ml_result["failure_probability"] * 100)

    return {
        "alert_type": alert_type,
        "severity": severity,
        "diagnosis": (
            f"Modelo de Machine Learning aponta {anomaly_pct}% de anomalia e {failure_pct}% de "
            f"probabilidade de falha. Causa provável: {reason}. "
            "Diagnóstico gerado automaticamente (agente de IA indisponível no momento)."
        ),
        "recommended_action": "Inspecionar o ativo e verificar o componente associado ao tipo de falha indicado.",
        "source": "rule-based-fallback",
    }
