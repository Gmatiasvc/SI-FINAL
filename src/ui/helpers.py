"""
Utilidades compartidas, caché de recursos/datos y funciones de formato para la interfaz
Streamlit de SolarGuard AI.

Paso a paso:
1. Provee `get_solarguard_graph()` cacheado con `@st.cache_resource` para reutilizar el
   grafo compilado LangGraph y su `MemorySaver` en toda la sesión sin recargar pesos.
2. Expone `load_json_metrics()` y `load_demo_presets()` cacheados con `@st.cache_data` para
   alimentar métricas, gráficos y casos demo locales desde el conjunto `test` real.
3. Provee manejo seguro de archivos cargados (`save_uploaded_file`), lectura en vivo de
   órdenes simuladas/logs diarios y generadores de reportes descargables (`.md` y `.json`).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.agents.graph import build_solarguard_graph
from src.core.config import (
    CACHE_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    ORDERS_DIR,
    SPLITS_DIR,
    VECTOR_STORE_DIR,
)
from src.core.logging_config import get_daily_log_path, get_logger

logger = get_logger(__name__)

UPLOADS_DIR = CACHE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource(max_entries=1)
def get_solarguard_graph() -> Any:
    """Compila y cachea una sola instancia del grafo MoE LangGraph con MemorySaver."""
    logger.info("[UI] Inicializando grafo SolarGuard AI en cache_resource...")
    return build_solarguard_graph()


@st.cache_data(ttl="5m", max_entries=16)
def load_json_metrics(filename: str) -> Dict[str, Any]:
    """Carga un archivo JSON de métricas desde `artifacts/metrics/`."""
    path = METRICS_DIR / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl="5m", max_entries=4)
def load_demo_presets() -> Dict[str, Dict[str, Any]]:
    """
    Construye los escenarios precargados para el Modo Demo Local usando ejemplos
    reales del conjunto `test` (sin fuga de datos).
    """
    rgb_csv = SPLITS_DIR / "rgb_splits.csv"
    ir_csv = SPLITS_DIR / "thermal_splits.csv"

    def _pick(csv_path: Path, label: str) -> str:
        if not csv_path.exists():
            return ""
        df = pd.read_csv(csv_path)
        sub = df[(df["split"] == "test") & (df["label"] == label)]
        if len(sub) == 0:
            sub = df[df["label"] == label]
        return str(sub.iloc[0]["image_path"]) if len(sub) > 0 else ""

    img_dirty = _pick(rgb_csv, "suciedad visible")
    img_crack = _pick(rgb_csv, "grieta visible")
    img_non_eval = _pick(rgb_csv, "vista no evaluable")
    img_ir_hotspot = _pick(ir_csv, "Hotspot / Celda crítica")

    return {
        "Manual": {
            "description": "Ingreso libre de telemetría e imágenes desde disco o cámara.",
            "panel_id": "INV-1BY6WEcLGh8j5v7",
            "fecha": "2020-06-15T12:00:00",
            "irradiancia": 0.75,
            "temp_amb": 28.5,
            "temp_mod": 46.0,
            "capacidad": 1191.07,
            "energia_observada": 940.0,
            "rgb_image_path": None,
            "thermal_image_path": None,
            "force_visual": False,
            "user_query": "",
        },
        "Escenario 1 · Normal (<15%)": {
            "description": "Pronóstico tabular dentro de línea base normal; omite visión y orden.",
            "panel_id": "INV-1BY6WEcLGh8j5v7",
            "fecha": "2020-06-15T12:00:00",
            "irradiancia": 0.75,
            "temp_amb": 28.5,
            "temp_mod": 46.0,
            "capacidad": 1191.07,
            "energia_observada": 945.0,
            "rgb_image_path": None,
            "thermal_image_path": None,
            "force_visual": False,
            "user_query": "¿Cómo proceder cuando la generación está dentro de lo esperado?",
        },
        "Escenario 2 · Suciedad RGB": {
            "description": "Inspección visual RGB de panel con suciedad visible, citas RAG y HITL.",
            "panel_id": "STRING-04-PANEL-12",
            "fecha": "2020-06-15T12:00:00",
            "irradiancia": 0.75,
            "temp_amb": 28.5,
            "temp_mod": 46.0,
            "capacidad": 1191.07,
            "energia_observada": 950.0,
            "rgb_image_path": img_dirty or None,
            "thermal_image_path": None,
            "force_visual": True,
            "user_query": "¿Qué controles manuales se recomiendan ante baja generación y suciedad visible?",
        },
        "Escenario 3 · Integral ML+RGB+IR": {
            "description": "Déficit energético >15%, grieta RGB, termografía IR y aprobación humana.",
            "panel_id": "STRING-07-PANEL-03",
            "fecha": "2020-06-15T12:00:00",
            "irradiancia": 0.75,
            "temp_amb": 28.5,
            "temp_mod": 46.0,
            "capacidad": 1191.07,
            "energia_observada": 590.0,
            "rgb_image_path": img_crack or None,
            "thermal_image_path": img_ir_hotspot or None,
            "force_visual": False,
            "user_query": "",
        },
        "Caso difícil · No evaluable": {
            "description": "Fotografía degradada o cubierta que dispara parada por incertidumbre.",
            "panel_id": "STRING-02-PANEL-09",
            "fecha": "2020-06-15T12:00:00",
            "irradiancia": 0.75,
            "temp_amb": 28.5,
            "temp_mod": 46.0,
            "capacidad": 1191.07,
            "energia_observada": 620.0,
            "rgb_image_path": img_non_eval or None,
            "thermal_image_path": None,
            "force_visual": False,
            "user_query": "",
        },
        "Consulta sin respaldo": {
            "description": "Consulta fuera de manuales para demostrar abstención segura sin alucinar.",
            "panel_id": "STRING-09-PANEL-01",
            "fecha": "2020-06-15T12:00:00",
            "irradiancia": 0.75,
            "temp_amb": 28.5,
            "temp_mod": 46.0,
            "capacidad": 1191.07,
            "energia_observada": 610.0,
            "rgb_image_path": img_dirty or None,
            "thermal_image_path": None,
            "force_visual": False,
            "user_query": "¿Cuál es la configuración Modbus TCP exacta para actualizar firmware Huawei SUN2000?",
        },
    }


def save_uploaded_file(uploaded_file: Any, prefix: str = "img") -> str:
    """Guarda un archivo subido en `artifacts/cache/uploads/` y devuelve su ruta absoluta."""
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    safe_name = f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
    out_path = UPLOADS_DIR / safe_name
    out_path.write_bytes(uploaded_file.getvalue())
    return str(out_path.resolve())


def get_system_health_status() -> List[Dict[str, Any]]:
    """Verifica la disponibilidad en disco de los 4 modelos entrenados y el índice FAISS."""
    checks = [
        ("Experto 1 · Tabular ML", MODELS_DIR / "tabular_best_model.joblib"),
        ("Experto 2A · YOLO Detector", MODELS_DIR / "yolo_panel_detector.pt"),
        ("Experto 2B · Clasificador RGB", MODELS_DIR / "rgb_classifier.pt"),
        ("Experto 3 · Termografía IR", MODELS_DIR / "thermal_classifier.pt"),
        ("Experto 4 · Índice FAISS RAG", VECTOR_STORE_DIR / "solarguard.faiss"),
    ]
    rows: List[Dict[str, Any]] = []
    for label, path in checks:
        exists = path.exists()
        size_mb = round(path.stat().st_size / (1024 * 1024), 2) if exists else 0.0
        rows.append({"component": label, "ready": exists, "size_mb": size_mb, "path": str(path)})
    return rows


def load_orders() -> List[Dict[str, Any]]:
    """Lee el registro persistente de órdenes simuladas (más recientes primero)."""
    orders_file = ORDERS_DIR / "revision_orders.json"
    if not orders_file.exists():
        return []
    try:
        data = json.loads(orders_file.read_text(encoding="utf-8"))
        return list(reversed(data)) if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def load_daily_log_lines(max_lines: int = 200) -> tuple[Path, List[str]]:
    """Lee las últimas `max_lines` líneas del archivo `.log` correspondiente al día actual."""
    log_path = get_daily_log_path()
    if not log_path.exists():
        return log_path, []
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return log_path, lines[-max_lines:]


def build_markdown_report(state_data: Dict[str, Any]) -> str:
    """Construye un reporte Markdown descargable a partir del estado estructurado del agente."""
    report = state_data.get("structured_report") or {}
    lines = [
        "# Reporte Técnico de Inspección — SolarGuard AI",
        "",
        f"- **Panel / Inversor:** {report.get('panel', state_data.get('panel_id', 'N/D'))}",
        f"- **Energía esperada:** {report.get('energia_esperada', 'N/D')}",
        f"- **Desviación energética:** {report.get('desviacion', 'N/D')}",
        f"- **Condición visual (RGB):** {report.get('condicion_visual', 'N/D')}",
    ]
    if report.get("condicion_termica"):
        lines.append(f"- **Condición térmica (IR):** {report.get('condicion_termica')}")
    lines.extend(
        [
            f"- **Incertidumbre activa:** {report.get('incertidumbre_activa', False)}",
            f"- **Estado de orden (HITL):** {report.get('estado_orden', 'SIN_ORDEN')}",
            "",
            "## Pasos de revisión recomendados",
        ]
    )
    for idx, step in enumerate(report.get("pasos_revision", []), start=1):
        lines.append(f"{idx}. {step}")

    lines.extend(["", "## Citas documentales verificables"])
    citas = report.get("citas", [])
    if citas:
        for cita in citas:
            lines.append(f"- {cita}")
    else:
        lines.append("- Sin respaldo documental aplicable.")

    lines.extend(["", "## Trazas de ejecución LangGraph"])
    for tr in state_data.get("trace_log", []):
        lines.append(f"- `{tr}`")

    return "\n".join(lines)


def build_json_report(state_data: Dict[str, Any]) -> str:
    """Serializa el diagnóstico y las trazas completas en JSON legible."""
    payload = {
        "panel_id": state_data.get("panel_id"),
        "expected_energy_kwh": state_data.get("expected_energy_kwh"),
        "energy_discrepancy_pct": state_data.get("energy_discrepancy_pct"),
        "is_anomaly": state_data.get("is_anomaly"),
        "rgb_classification": state_data.get("rgb_classification"),
        "thermal_classification": state_data.get("thermal_classification"),
        "structured_report": state_data.get("structured_report"),
        "order_record": state_data.get("order_record"),
        "trace_log": state_data.get("trace_log", []),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def get_figure_path(filename: str) -> Path | None:
    """Devuelve la ruta a una figura generada en `report/figures/` si existe."""
    p = FIGURES_DIR / filename
    return p if p.exists() else None
