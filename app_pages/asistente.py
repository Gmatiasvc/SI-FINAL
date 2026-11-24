"""
Página del Asistente Documental RAG Multi-turno.

Paso a paso:
1. Lee el último diagnóstico almacenado en `st.session_state.last_agent_state` (si existe)
   para contextualizar las respuestas sin re-ejecutar innecesariamente los modelos ML/CV.
2. Muestra el historial de conversación con `st.chat_message` y chips de sugerencia iniciales
   (`st.pills` en `st.empty()`) cuando aún no hay mensajes.
3. Procesa cada nueva consulta mostrando una línea de tiempo compacta (`type="compact"` con
   `type="step"` anidados): Guardrails anti-Prompt Injection -> Búsqueda FAISS/TF-IDF -> Síntesis
   fundamentada con citas `(documento, página)` o abstención explícita si no hay respaldo.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import streamlit as st

from src.core.logging_config import get_logger
from src.rag.guardrails import inspect_prompt_injection
from src.rag.retriever import consultar_manual_solar

logger = get_logger(__name__)

SUGGESTIONS = {
    ":material/help: Controles ante suciedad": "¿Qué controles manuales se recomiendan ante baja generación y suciedad visible?",
    ":material/cleaning_services: Protocolo de limpieza": "¿Cómo se deben limpiar los paneles solares con polvo o excrementos de aves sin dañar el vidrio?",
    ":material/thermostat: Criterios Hotspot IR": "¿Qué diferencial térmico Delta-T caracteriza un hotspot o celda crítica y qué revisión hacer?",
    ":material/warning: Prueba sin respaldo": "¿Cuál es la configuración Modbus TCP exacta para actualizar firmware Huawei SUN2000?",
}


def _build_grounded_answer(
    query: str,
    rag_res: Dict[str, Any],
    case_state: Dict[str, Any] | None,
) -> str:
    """Construye una respuesta transparente diferenciando diagnóstico del caso y citas RAG."""
    parts: List[str] = []

    if rag_res.get("guardrail_alert") and not rag_res.get("has_evidence"):
        parts.append(
            f"**Bloqueo de seguridad:** {rag_res['status_message']}\n\n"
            f"Detalle: {rag_res['guardrail_alert']}"
        )
        return "\n".join(parts)

    if case_state and case_state.get("structured_report"):
        rep = case_state["structured_report"]
        parts.append(
            f"**Contexto del caso activo (`{rep.get('panel')}`):** "
            f"Energía esperada `{rep.get('energia_esperada')}`, "
            f"desviación `{rep.get('desviacion')}`, "
            f"condición visual `{rep.get('condicion_visual')}`."
        )

    if not rag_res.get("has_evidence"):
        parts.append(
            f"**Sin respaldo documental suficiente:** {rag_res.get('status_message')}\n\n"
            "No se inventan procedimientos ni referencias fuera del corpus técnico indexado."
        )
        return "\n".join(parts)

    parts.append("**Evidencia documental recuperada:**")
    for doc in rag_res["retrieved_docs"]:
        parts.append(
            f"- **{doc['source']} (p. {doc['page']}, score {doc['score']:.3f}):** {doc['content']}"
        )

    parts.append(
        "\n*Nota operativa: Toda maniobra eléctrica o aislamiento LOTO debe ser ejecutada "
        "manualmente por un técnico habilitado; no se emiten desconexiones automáticas.*"
    )
    return "\n".join(parts)


# Mostrar resumen del contexto actual si existe una inspección previa.
active_case = st.session_state.get("last_agent_state")
if active_case and active_case.get("structured_report"):
    rep_ctx = active_case["structured_report"]
    st.info(
        f"Panel `{rep_ctx.get('panel')}` · Visual: `{rep_ctx.get('condicion_visual')}` · "
        f"Desviación: `{rep_ctx.get('desviacion')}`",
        title="Contexto de inspección cargado",
        icon=":material/solar_power:",
    )
else:
    st.caption(
        "Puedes consultar directamente los manuales fotovoltaicos indexados o ejecutar primero "
        "una inspección en **Inspección integral** para contextualizar las respuestas."
    )

# Renderizar historial previo de conversación.
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            st.caption("Citas: " + " · ".join(msg["citations"]))

# Entrada de chat y sugerencias iniciales.
prompt = st.chat_input(
    "Escribe una consulta técnica o prueba una pregunta sin respaldo / adversarial...",
    submit_mode="disable",
    key="rag_chat_input",
)

if not st.session_state.chat_messages:
    suggestions_box = st.empty()
    selected_chip = suggestions_box.pills(
        "Consultas de ejemplo",
        list(SUGGESTIONS.keys()),
        label_visibility="collapsed",
        key="rag_suggestion_pills",
    )
    if selected_chip:
        prompt = SUGGESTIONS[selected_chip]
    if prompt:
        suggestions_box.empty()

if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        with st.status(":shimmer[Consultando manuales y guardrails]", type="compact") as status:
            with st.status("Verificando guardrails anti-Prompt Injection", type="step") as s1:
                guard_chk = inspect_prompt_injection(prompt, source_label="streamlit_chat")
                st.write(guard_chk.reason)
                s1.update(
                    state="complete" if guard_chk.is_safe else "error",
                    label=(
                        "Guardrails: consulta segura"
                        if guard_chk.is_safe
                        else "Guardrails: patrón adversarial bloqueado"
                    ),
                )

            with st.status("Recuperando fragmentos en FAISS + TF-IDF", type="step") as s2:
                rag_res = consultar_manual_solar(prompt, top_k=3)
                st.write(rag_res["status_message"])
                s2.update(state="complete")

            elapsed = round((time.perf_counter() - t0) * 1000.0, 1)
            status.update(
                label=f"Completado en {elapsed:.1f} ms",
                state="complete",
            )

        answer = _build_grounded_answer(prompt, rag_res, active_case)
        st.markdown(answer)
        if rag_res.get("citations"):
            st.caption("Citas: " + " · ".join(rag_res["citations"]))

    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": rag_res.get("citations", []),
        }
    )
    logger.info(
        f"[UI-CHAT] Consulta procesada has_evidence={rag_res.get('has_evidence')} "
        f"citas={rag_res.get('citations')}"
    )
