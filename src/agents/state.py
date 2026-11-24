"""
Definición del estado tipado `SolarGuardState` para el grafo MoE en LangGraph.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class SolarGuardState(TypedDict, total=False):
    """
    Estado compartido que fluye entre todos los nodos del grafo SolarGuard AI.
    Cada experto lee únicamente las entradas necesarias y escribe sus hallazgos trazables.
    """

    # --- Entradas del caso ---
    panel_id: str
    telemetry: Dict[str, Any]  # irradiancia, temp_amb, temp_mod, capacidad, energia_observada, fecha
    rgb_image_path: Optional[str]
    thermal_image_path: Optional[str]
    user_query: Optional[str]
    force_visual_inspection: bool

    # --- Resultados de Expertos ---
    expected_energy_kwh: Optional[float]
    expected_uncertainty_kwh: Optional[float]
    energy_discrepancy_pct: Optional[float]
    signed_deviation_pct: Optional[float]
    is_anomaly: bool
    detected_panels_crops: List[str]
    rgb_classification: Optional[Dict[str, Any]]      # {label, confidence, all_probs, es_evaluable}
    thermal_classification: Optional[Dict[str, Any]]  # {label, confidence, all_probs}

    # --- RAG y Salida Estructurada ---
    retrieved_docs: List[Dict[str, Any]]              # {chunk_id, content, source, page, score}
    rag_has_evidence: bool
    rag_status_message: str
    structured_report: Optional[Dict[str, Any]]       # Validado por Pydantic

    # --- Control, Trazabilidad y Human-in-the-Loop ---
    uncertainty_flag: bool
    uncertainty_reason: Optional[str]
    human_approval_required: bool
    human_approved: Optional[bool]
    human_reviewer_note: Optional[str]
    order_record: Optional[Dict[str, Any]]
    guardrail_warnings: List[str]
    trace_log: List[str]
    errors: List[str]
