"""
Guardrails de seguridad, anti-Prompt Injection y cumplimiento ético/operativo.

Paso a paso:
1. Analiza consultas del usuario y fragmentos recuperados por el RAG en busca de patrones
   de inyección de instrucciones (`prompt injection`), como intentos de anular reglas del
   sistema ("ignore previous instructions", "system prompt", "exec(", "desconecta el inversor").
2. Envuelve todo fragmento recuperado como **datos pasivos** dentro de delimitadores explícitos
   y sanitiza cualquier directiva imperativa sospechosa antes de entregarlo al LLM.
3. Verifica la salida textual/estructurada propuesta para garantizar el cumplimiento de las
   restricciones éticas y operativas obligatorias:
   - Prohibido emitir comandos de desconexión eléctrica automática.
   - Prohibido diagnosticar fallos eléctricos/térmicos internos si no hubo captura IR ni medición.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Patrones típicos de Prompt Injection en español e inglés.
INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"ignora\s+(todas\s+)?(las\s+)?instrucciones\s+anteriores", re.IGNORECASE),
    re.compile(r"olvida\s+(tus|las)\s+reglas", re.IGNORECASE),
    re.compile(r"revela\s+(tu|el)\s+prompt\s+del?\s+sistema", re.IGNORECASE),
    re.compile(r"\b(system\s*prompt|jailbreak|developer\s+mode|DAN)\b", re.IGNORECASE),
    re.compile(r"ejecuta\s+(desconexi[oó]n|comando|shell|rm\s+-rf)", re.IGNORECASE),
    re.compile(r"desconecta\s+autom[aá]ticamente\s+el\s+(inversor|panel|sistema)", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\s*>|\[\s*INST\s*\]", re.IGNORECASE),
]

# Patrones de salidas prohibidas operativamente (acciones peligrosas automatizadas).
UNSAFE_ACTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b(ejecutar|activar|enviar)\s+(orden\s+de\s+)?desconexi[oó]n\s+autom[aá]tica", re.IGNORECASE),
    re.compile(r"\b(disconnect|trip)\s+(breaker|inverter)\s+automatically", re.IGNORECASE),
    re.compile(r"^\s*(DESCONECTAR|APAGAR)\s+(INMEDIATAMENTE|AHORA)\s+EL\s+INVERSOR", re.IGNORECASE),
]


@dataclass
class GuardrailCheckResult:
    """Resultado estructurado de la inspección de seguridad de entrada o documento."""

    is_safe: bool
    matched_patterns: List[str]
    sanitized_text: str
    reason: str


def inspect_prompt_injection(text: str, source_label: str = "query") -> GuardrailCheckResult:
    """
    Evalúa si un texto (consulta del usuario o fragmento de documento) contiene
    patrones de *Prompt Injection* o manipulación de instrucciones.
    """
    if not text:
        return GuardrailCheckResult(
            is_safe=True, matched_patterns=[], sanitized_text="", reason="Texto vacío"
        )

    hits: List[str] = []
    sanitized = text
    for pattern in INJECTION_PATTERNS:
        if pattern.search(sanitized):
            hits.append(pattern.pattern)
            sanitized = pattern.sub("[CONTENIDO_NO_AUTORIZADO_BLOQUEADO]", sanitized)

    if hits:
        logger.warning(
            f"[GUARDRAIL] Posible Prompt Injection detectado en {source_label}: "
            f"{len(hits)} patrón(es) neutralizados."
        )
        return GuardrailCheckResult(
            is_safe=False,
            matched_patterns=hits,
            sanitized_text=sanitized,
            reason=(
                "Se detectaron instrucciones potencialmente maliciosas o contrarias a las "
                "políticas de seguridad; los fragmentos recuperados se tratan solo como datos pasivos."
            ),
        )

    return GuardrailCheckResult(
        is_safe=True,
        matched_patterns=[],
        sanitized_text=sanitized,
        reason="Limpio",
    )


def wrap_passive_rag_context(docs: List[dict]) -> Tuple[str, bool]:
    """
    Formatea los fragmentos recuperados del RAG como datos estrictamente pasivos,
    neutralizando cualquier posible instrucción incrustada dentro de un documento.

    Retorna `(context_str, had_injection_in_docs)`.
    """
    had_injection = False
    blocks: List[str] = []
    for idx, d in enumerate(docs, start=1):
        raw_content = str(d.get("content", ""))
        check = inspect_prompt_injection(raw_content, source_label=f"rag_doc_{idx}")
        if not check.is_safe:
            had_injection = True
        blocks.append(
            f"[DATO_PASIVO_INICIO id={idx} fuente='{d.get('source')}' pagina={d.get('page')} "
            f"score={d.get('score')}]\n"
            f"{check.sanitized_text}\n"
            f"[DATO_PASIVO_FIN id={idx}]"
        )
    return "\n\n".join(blocks), had_injection


def validate_operational_safety(
    pasos_revision: List[str],
    has_thermal_evidence: bool,
) -> Tuple[List[str], List[str]]:
    """
    Audita la lista de pasos propuestos antes de enviarla al técnico humano:
    1. Bloquea cualquier instrucción de desconexión eléctrica automatizada.
    2. Añade advertencia explícita si no existe evidencia térmica (solo análisis RGB).
    """
    warnings: List[str] = []
    safe_steps: List[str] = []

    for step in pasos_revision:
        blocked = any(pat.search(step) for pat in UNSAFE_ACTION_PATTERNS)
        if blocked:
            msg = (
                "Se bloqueó una recomendación de desconexión eléctrica automatizada "
                "conforme a la política de control humano obligatorio."
            )
            logger.warning(f"[GUARDRAIL] {msg} Paso original='{step}'")
            warnings.append(msg)
            continue
        safe_steps.append(step)

    if not has_thermal_evidence:
        note = (
            "Alcance sin termografía: no se infieren fallos eléctricos o térmicos internos "
            "sin medición térmica IR o verificación eléctrica presencial con instrumentación."
        )
        if not any("sin medición" in s.lower() or "termografía" in s.lower() for s in safe_steps):
            safe_steps.append(note)

    return safe_steps, warnings
