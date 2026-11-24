"""
Esquemas Pydantic para validar entradas, salidas estructuradas del LLM y órdenes simuladas.

Garantiza el contrato obligatorio de salida del asistente:
`{panel, energia_esperada, desviacion, condicion_visual, pasos_revision, citas}`
extendido con trazabilidad de termografía condicional, incertidumbre y aprobación humana.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class StructuredInspectionReport(BaseModel):
    """Contrato estructurado validado por Pydantic para toda recomendación de SolarGuard AI."""

    panel: str = Field(..., description="Identificador del panel o inversor evaluado.")
    energia_esperada: str = Field(
        ..., description="Energía esperada por el modelo tabular con banda de incertidumbre."
    )
    desviacion: str = Field(
        ..., description="Discrepancia porcentual frente a la energía real observada."
    )
    condicion_visual: str = Field(
        ..., description="Diagnóstico RGB (4 clases) y nivel de confianza."
    )
    condicion_termica: Optional[str] = Field(
        default=None,
        description="Resultado térmico IR condicional si se adjuntó captura termográfica.",
    )
    pasos_revision: List[str] = Field(
        default_factory=list,
        description="Pasos de inspección técnica verificables sin comandos de desconexión.",
    )
    citas: List[str] = Field(
        default_factory=list,
        description="Referencias documentales recuperadas con fuente y página.",
    )
    incertidumbre_activa: bool = Field(
        default=False,
        description="Indica si se detuvo el diagnóstico por vista no evaluable o baja confianza.",
    )
    requiere_aprobacion_humana: bool = Field(
        default=False,
        description="Indica si se requiere validación del técnico humano antes de emitir orden.",
    )
    estado_orden: str = Field(
        default="SIN_ORDEN",
        description="Estado del flujo HITL: SIN_ORDEN, PENDIENTE_APROBACION, APROBADA, RECHAZADA.",
    )
    respaldo_documental: bool = Field(
        default=True,
        description="False cuando el RAG no encontró evidencia suficiente en el corpus.",
    )
    advertencias_seguridad: List[str] = Field(
        default_factory=list,
        description="Alertas emitidas por los guardrails de seguridad o Prompt Injection.",
    )


class RevisionOrderRecord(BaseModel):
    """Registro persistente de una orden de revisión técnica simulada (sin maniobra eléctrica)."""

    order_id: str
    created_at_utc: str
    panel_id: str
    priority: str
    diagnosis_summary: str
    pasos_revision: List[str]
    citas: List[str]
    approved_by_human: bool = True
    reviewer_note: str = ""
    status: str = "EMITIDA_SIMULADA"
