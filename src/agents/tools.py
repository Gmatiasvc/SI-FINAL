"""
Herramientas funcionales ejecutables del agente SolarGuard AI (LangChain `@tool`).

Implementa las 4 herramientas exigidas en el diseño:
1. `predecir_generacion(fecha, condiciones)`
2. `clasificar_panel(ruta_imagen)`
3. `consultar_manual_solar(consulta)`
4. `simular_orden_revision(panel, ...)`

Cada herramienta ejecuta lógica real sobre los modelos entrenados, el índice FAISS o el
registro persistente de órdenes en `artifacts/orders/revision_orders.json`, y registra su
invocación en los logs con colores y archivo diario.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from src.agents.schemas import RevisionOrderRecord
from src.core.config import ORDERS_DIR
from src.core.logging_config import get_logger
from src.core.telemetry import track_event
from src.ml.predict_tabular import predecir_generacion as _ml_predict
from src.rag.guardrails import validate_operational_safety
from src.rag.retriever import consultar_manual_solar as _rag_search
from src.vision.inference import clasificar_panel as _cv_classify_rgb

logger = get_logger(__name__)

ORDERS_FILE = ORDERS_DIR / "revision_orders.json"


def run_predecir_generacion(
    fecha: str,
    condiciones: Dict[str, Any],
    energia_observada: Optional[float] = None,
    panel_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Envoltorio directo para predecir generación esperada y discrepancia energética."""
    logger.info(f"[TOOL] Invocando predecir_generacion(fecha={fecha}, panel_id={panel_id})")
    return _ml_predict(
        fecha=fecha,
        condiciones=condiciones,
        energia_observada=energia_observada,
        panel_id=panel_id,
    )


def run_clasificar_panel(ruta_imagen: str, run_yolo: bool = True) -> Dict[str, Any]:
    """Envoltorio directo para detección espacial YOLO + clasificación visual RGB."""
    logger.info(f"[TOOL] Invocando clasificar_panel(ruta_imagen={ruta_imagen})")
    return _cv_classify_rgb(ruta_imagen=ruta_imagen, run_yolo=run_yolo)


def run_consultar_manual_solar(consulta: str, top_k: int = 3) -> Dict[str, Any]:
    """Envoltorio directo para recuperación documental verificable con guardrails."""
    logger.info(f"[TOOL] Invocando consultar_manual_solar(consulta='{consulta[:70]}...')")
    return _rag_search(consulta=consulta, top_k=top_k)


def run_simular_orden_revision(
    panel: str,
    diagnostico: str = "Discrepancia energética o anomalía visual detectada",
    pasos_revision: Optional[List[str]] = None,
    citas: Optional[List[str]] = None,
    prioridad: str = "MEDIA",
    approved_by_human: bool = True,
    reviewer_note: str = "",
    has_thermal_evidence: bool = False,
    orders_path: Path = ORDERS_FILE,
) -> Dict[str, Any]:
    """
    Simula y persiste una orden de revisión técnica en disco (SIN desconexión eléctrica).
    """
    with track_event(logger, "tool_simular_orden_revision", panel=panel, prioridad=prioridad) as ctx:
        if not approved_by_human:
            raise PermissionError(
                "No se puede emitir la orden de revisión sin aprobación humana explícita."
            )

        raw_steps = pasos_revision or [
            "Realizar inspección visual presencial en campo con EPP dieléctrico.",
            "Medir tensión Voc y corriente de string en caja combinadora antes de intervenir.",
        ]
        safe_steps, safety_warnings = validate_operational_safety(
            raw_steps, has_thermal_evidence=has_thermal_evidence
        )

        order = RevisionOrderRecord(
            order_id=f"ORD-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            panel_id=panel,
            priority=prioridad,
            diagnosis_summary=diagnostico,
            pasos_revision=safe_steps,
            citas=citas or [],
            approved_by_human=approved_by_human,
            reviewer_note=reviewer_note,
            status="EMITIDA_SIMULADA",
        )

        orders_path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, Any]] = []
        if orders_path.exists():
            try:
                existing = json.loads(orders_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = []

        order_dict = order.model_dump()
        if safety_warnings:
            order_dict["safety_warnings"] = safety_warnings
        existing.append(order_dict)
        orders_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        ctx["order_id"] = order.order_id
        logger.info(
            f"[TOOL] Orden simulada persistida: id={order.order_id}, panel={panel}, "
            f"prioridad={prioridad}, archivo={orders_path}"
        )
        return order_dict


# ------------------------------------------------------------------------------
# Definición oficial de herramientas LangChain (@tool) para integración con LLM
# ------------------------------------------------------------------------------
@tool("predecir_generacion")
def predecir_generacion(fecha: str, condiciones: Dict[str, Any]) -> Dict[str, Any]:
    """Predice energia_kwh_hora esperada según irradiancia, temperaturas, capacidad y fecha ISO."""
    obs = condiciones.get("energia_observada")
    panel_id = condiciones.get("panel_id")
    return run_predecir_generacion(
        fecha=fecha, condiciones=condiciones, energia_observada=obs, panel_id=panel_id
    )


@tool("clasificar_panel")
def clasificar_panel(ruta_imagen: str) -> Dict[str, Any]:
    """Detecta paneles con YOLO y clasifica su estado visual RGB en 4 clases con confianza."""
    return run_clasificar_panel(ruta_imagen=ruta_imagen, run_yolo=True)


@tool("consultar_manual_solar")
def consultar_manual_solar(consulta: str) -> Dict[str, Any]:
    """Busca protocolos de mantenimiento y seguridad FV en manuales indexados con citas exactas."""
    return run_consultar_manual_solar(consulta=consulta, top_k=3)


@tool("simular_orden_revision")
def simular_orden_revision(panel: str) -> Dict[str, Any]:
    """Genera una orden de inspección técnica simulada (prohibido desconectar equipos)."""
    return run_simular_orden_revision(panel=panel, approved_by_human=True)


SOLARGUARD_TOOLS = [
    predecir_generacion,
    clasificar_panel,
    consultar_manual_solar,
    simular_orden_revision,
]
