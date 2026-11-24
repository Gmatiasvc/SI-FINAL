"""
Ejecutor CLI integral de escenarios de demostración y auditoría end-to-end para SolarGuard AI.

Ejecuta sin interfaz Streamlit los escenarios obligatorios del proyecto:
1. Escenario 1: Ruta normal (discrepancia < 15%, sin inspección visual).
2. Escenario 2: Fotografía RGB con suciedad/daño visible + RAG + pausa/aprobación HITL.
3. Escenario 3: Discrepancia energética >= 15% + RGB + Termografía IR condicional + RAG + HITL.
4. Caso difícil: Imagen degradada (`vista no evaluable`) -> parada segura por incertidumbre.
5. Caso de error controlado: Irradiancia negativa -> ruta `manejar_error`.
6. Caso sin respaldo / Prompt Injection -> verificación de abstención y guardrails.

Exporta los reportes resultantes en `artifacts/reports/` y registra latencia y trazas completas.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.agents.graph import build_solarguard_graph
from src.core.config import REPORTS_DIR, SPLITS_DIR
from src.core.logging_config import get_logger, log_section
from src.core.telemetry import log_metric, track_event

logger = get_logger(__name__)


def _pick_sample_image(splits_csv: Path, label: str) -> str:
    """Selecciona una imagen real del conjunto `test` para pruebas integradas sin fuga."""
    df = pd.read_csv(splits_csv)
    sub = df[(df["split"] == "test") & (df["label"] == label)]
    if len(sub) == 0:
        sub = df[df["label"] == label]
    return str(sub.iloc[0]["image_path"])


def _run_scenario(
    graph: Any,
    scenario_id: str,
    initial_state: Dict[str, Any],
    auto_approve_hitl: bool = True,
    reviewer_note: str = "Validación técnica aprobada en ejecución de prueba.",
) -> Dict[str, Any]:
    """Ejecuta un escenario en el grafo LangGraph, manejando la pausa HITL si ocurre."""
    t0 = time.perf_counter()
    config = {"configurable": {"thread_id": f"{scenario_id}-{uuid.uuid4().hex[:6]}"}}
    state_out = graph.invoke(initial_state, config=config)

    snapshot = graph.get_state(config)
    interrupted_for_hitl = bool(snapshot.next)

    if interrupted_for_hitl and "validar_tecnico" in snapshot.next:
        logger.info(
            f"[HITL] Grafo pausado antes de `validar_tecnico` para {scenario_id}. "
            f"Decisión simulada del técnico: approve={auto_approve_hitl}"
        )
        graph.update_state(
            config,
            {
                "human_approved": auto_approve_hitl,
                "human_reviewer_note": reviewer_note,
            },
        )
        state_out = graph.invoke(None, config=config)

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    result = {
        "scenario_id": scenario_id,
        "latency_ms": latency_ms,
        "interrupted_for_hitl": interrupted_for_hitl,
        "uncertainty_flag": bool(state_out.get("uncertainty_flag", False)),
        "is_anomaly": bool(state_out.get("is_anomaly", False)),
        "order_id": (state_out.get("order_record") or {}).get("order_id"),
        "structured_report": state_out.get("structured_report"),
        "trace_log": state_out.get("trace_log", []),
        "errors": state_out.get("errors", []),
    }
    return result


def run_all_scenarios() -> Dict[str, Any]:
    """Ejecuta todos los escenarios end-to-end y guarda un reporte maestro JSON + Markdown."""
    log_section(logger, "EJECUCIÓN INTEGRAL DE ESCENARIOS LANGGRAPH (MoE + HITL)")
    graph = build_solarguard_graph()

    rgb_splits = SPLITS_DIR / "rgb_splits.csv"
    ir_splits = SPLITS_DIR / "thermal_splits.csv"

    img_dirty = _pick_sample_image(rgb_splits, "suciedad visible")
    img_crack = _pick_sample_image(rgb_splits, "grieta visible")
    img_non_eval = _pick_sample_image(rgb_splits, "vista no evaluable")
    img_ir_hotspot = _pick_sample_image(ir_splits, "Hotspot / Celda crítica")

    scenarios: List[Dict[str, Any]] = []

    with track_event(logger, "run_all_agent_scenarios") as ctx:
        # Escenario 1: Generación normal (discrepancia < 15%)
        scenarios.append(
            _run_scenario(
                graph,
                "escenario_1_normal",
                {
                    "panel_id": "INV-1BY6WEcLGh8j5v7",
                    "telemetry": {
                        "fecha": "2020-06-15T12:00:00",
                        "irradiancia": 0.75,
                        "temp_amb": 28.5,
                        "temp_mod": 46.0,
                        "energia_observada": 940.0,
                    },
                    "user_query": "¿Cómo proceder cuando la generación está dentro de lo esperado?",
                },
            )
        )

        # Escenario 2: Fotografía RGB con suciedad visible + RAG + HITL aprobado
        scenarios.append(
            _run_scenario(
                graph,
                "escenario_2_vision_rgb_suciedad",
                {
                    "panel_id": "STRING-04-PANEL-12",
                    "telemetry": {
                        "fecha": "2020-06-15T12:00:00",
                        "irradiancia": 0.75,
                        "temp_amb": 28.5,
                        "temp_mod": 46.0,
                        "energia_observada": 950.0,
                    },
                    "rgb_image_path": img_dirty,
                    "force_visual_inspection": True,
                    "user_query": "¿Cómo limpiar paneles con suciedad visible o excrementos?",
                },
                auto_approve_hitl=True,
                reviewer_note="Limpieza programada para mañana 06:00 AM.",
            )
        )

        # Escenario 3: Discrepancia >= 15% + RGB grieta + Termografía IR + RAG + HITL aprobado
        scenarios.append(
            _run_scenario(
                graph,
                "escenario_3_integral_ml_rgb_ir",
                {
                    "panel_id": "STRING-07-PANEL-03",
                    "telemetry": {
                        "fecha": "2020-06-15T12:00:00",
                        "irradiancia": 0.75,
                        "temp_amb": 28.5,
                        "temp_mod": 46.0,
                        "energia_observada": 590.0,  # ~39% de déficit frente a ~970 kWh esperados
                    },
                    "rgb_image_path": img_crack,
                    "thermal_image_path": img_ir_hotspot,
                },
                auto_approve_hitl=True,
                reviewer_note="Confirmada grieta + hotspot; cuadrilla despachada con Megger.",
            )
        )

        # Caso difícil: Vista no evaluable -> parada segura por incertidumbre
        scenarios.append(
            _run_scenario(
                graph,
                "caso_dificil_vista_no_evaluable",
                {
                    "panel_id": "STRING-02-PANEL-09",
                    "telemetry": {
                        "fecha": "2020-06-15T12:00:00",
                        "irradiancia": 0.75,
                        "temp_amb": 28.5,
                        "temp_mod": 46.0,
                        "energia_observada": 620.0,
                    },
                    "rgb_image_path": img_non_eval,
                },
            )
        )

        # Caso de error controlado: irradiancia física negativa
        scenarios.append(
            _run_scenario(
                graph,
                "caso_error_entrada_invalida",
                {
                    "panel_id": "INV-ERR-01",
                    "telemetry": {
                        "fecha": "2020-06-15T12:00:00",
                        "irradiancia": -0.5,
                        "temp_amb": 28.5,
                        "temp_mod": 46.0,
                        "energia_observada": 500.0,
                    },
                },
            )
        )

        # Caso sin respaldo documental y rechazo humano
        scenarios.append(
            _run_scenario(
                graph,
                "caso_sin_respaldo_y_rechazo_hitl",
                {
                    "panel_id": "STRING-09-PANEL-01",
                    "telemetry": {
                        "fecha": "2020-06-15T12:00:00",
                        "irradiancia": 0.75,
                        "temp_amb": 28.5,
                        "temp_mod": 46.0,
                        "energia_observada": 610.0,
                    },
                    "rgb_image_path": img_dirty,
                    "user_query": "¿Cuál es la configuración Modbus TCP exacta para actualizar firmware Huawei?",
                },
                auto_approve_hitl=False,
                reviewer_note="Rechazada por falta de manual específico de firmware.",
            )
        )

        ctx["n_scenarios"] = len(scenarios)

    summary_payload = {
        "total_scenarios": len(scenarios),
        "scenarios": scenarios,
    }
    json_out = REPORTS_DIR / "scenarios_execution_report.json"
    json_out.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Generar también un resumen Markdown legible de las trazas y reportes emitidos.
    md_lines = ["# Reporte de Escenarios Ejecutados — SolarGuard AI\n"]
    for sc in scenarios:
        rep = sc.get("structured_report") or {}
        md_lines.append(f"## {sc['scenario_id']} ({sc['latency_ms']} ms)")
        md_lines.append(f"- **Panel:** {rep.get('panel')}")
        md_lines.append(f"- **Energía esperada:** {rep.get('energia_esperada')}")
        md_lines.append(f"- **Desviación:** {rep.get('desviacion')}")
        md_lines.append(f"- **Condición visual:** {rep.get('condicion_visual')}")
        if rep.get("condicion_termica"):
            md_lines.append(f"- **Condición térmica:** {rep.get('condicion_termica')}")
        md_lines.append(f"- **Estado orden HITL:** {rep.get('estado_orden')}")
        md_lines.append(f"- **Citas RAG:** {rep.get('citas')}")
        md_lines.append("- **Trazas LangGraph:**")
        for tr in sc.get("trace_log", []):
            md_lines.append(f"  - `{tr}`")
        md_lines.append("")

    md_out = REPORTS_DIR / "scenarios_execution_report.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    log_metric(
        logger,
        "agent.e2e_scenarios",
        {
            "total": len(scenarios),
            "avg_latency_ms": round(sum(s["latency_ms"] for s in scenarios) / len(scenarios), 2),
        },
    )
    logger.info(f"Reportes de escenarios guardados en {json_out} y {md_out}")
    return summary_payload


if __name__ == "__main__":
    run_all_scenarios()
