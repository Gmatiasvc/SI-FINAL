"""
Página principal de Inspección Integral (Orquestación MoE LangGraph + Human-in-the-Loop).

Paso a paso:
1. Permite seleccionar un escenario del Modo Demo Local (`st.segmented_control`) que
   precarga telemetría, fotografías reales del split `test` y consultas de prueba.
2. Captura variables meteorológicas/operativas e imágenes opcionales (RGB y Termografía IR)
   dentro de un `st.form` para evitar re-ejecuciones innecesarias antes de enviar.
3. Invoca el grafo LangGraph cacheado (`get_solarguard_graph()`), midiendo latencia real y
   detectando si se pausó en `interrupt_before=["validar_tecnico"]`.
4. Presenta métricas KPI, diagnóstico visual (con recortes YOLO y probabilidades), resultado
   térmico condicional, citas RAG verificables, panel HITL de aprobación/rechazo, línea de
   tiempo por nodo y botones de descarga (`.md` y `.json`).
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.core.config import ENERGY_DISCREPANCY_THRESHOLD_PCT, VISION_CONFIDENCE_THRESHOLD
from src.core.logging_config import get_logger
from src.ui.helpers import (
    build_json_report,
    build_markdown_report,
    get_solarguard_graph,
    load_demo_presets,
    save_uploaded_file,
)

logger = get_logger(__name__)
presets = load_demo_presets()


def _on_preset_change() -> None:
    """Actualiza los campos del formulario cuando el usuario elige otro escenario demo."""
    selected = st.session_state.get("demo_preset_control", "Manual")
    data = presets.get(selected, presets["Manual"])
    st.session_state.inp_panel_id = data["panel_id"]
    st.session_state.inp_fecha = data["fecha"]
    st.session_state.inp_irradiancia = float(data["irradiancia"])
    st.session_state.inp_temp_amb = float(data["temp_amb"])
    st.session_state.inp_temp_mod = float(data["temp_mod"])
    st.session_state.inp_capacidad = float(data["capacidad"])
    st.session_state.inp_energia_observada = float(data["energia_observada"])
    st.session_state.inp_user_query = data["user_query"]
    st.session_state.inp_force_visual = bool(data["force_visual"])
    st.session_state.preset_rgb_path = data["rgb_image_path"]
    st.session_state.preset_thermal_path = data["thermal_image_path"]


# ==============================================================================
# 1. Selector de Modo Demo Local (Offline)
# ==============================================================================
st.caption(
    "Integra predicción tabular (`energia_kwh_hora`), detección YOLOv8n + clasificación RGB, "
    "termografía IR condicional, manuales RAG y aprobación humana."
)

selected_preset = st.segmented_control(
    "Modo demo local (escenarios precargados del conjunto test)",
    options=list(presets.keys()),
    key="demo_preset_control",
    on_change=_on_preset_change,
    width="stretch",
)
active_preset_name = selected_preset or "Manual"
st.caption(f":material/info: {presets[active_preset_name]['description']}")

# ==============================================================================
# 2. Formulario de entrada tabular y capturas visuales
# ==============================================================================
with st.form("inspection_form", border=True):
    st.subheader("Parámetros de entrada e imágenes", icon=":material/science:")
    c1, c2, c3 = st.columns(3)
    with c1:
        panel_id = st.text_input("Identificador de panel / inversor", key="inp_panel_id")
    with c2:
        fecha = st.text_input("Fecha y hora (ISO-8601)", key="inp_fecha")
    with c3:
        capacidad = st.number_input(
            "Capacidad instalada (kWh/h)",
            min_value=1.0,
            max_value=50000.0,
            step=10.0,
            key="inp_capacidad",
        )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        irradiancia = st.number_input(
            "Irradiancia normalizada",
            min_value=-1.0,
            max_value=2.5,
            step=0.05,
            key="inp_irradiancia",
            help="Rango físico válido >= 0. Valores negativos prueban la ruta de error.",
        )
    with m2:
        temp_amb = st.number_input(
            "Temp. ambiente (°C)",
            min_value=-30.0,
            max_value=65.0,
            step=0.5,
            key="inp_temp_amb",
        )
    with m3:
        temp_mod = st.number_input(
            "Temp. de módulo (°C)",
            min_value=-30.0,
            max_value=105.0,
            step=0.5,
            key="inp_temp_mod",
        )
    with m4:
        energia_observada = st.number_input(
            "Energía observada (kWh/h)",
            min_value=0.0,
            max_value=50000.0,
            step=10.0,
            key="inp_energia_observada",
        )

    u1, u2 = st.columns(2)
    with u1:
        rgb_upload = st.file_uploader(
            "Fotografía RGB del panel (opcional)",
            type=["jpg", "jpeg", "png", "webp"],
            key="uploader_rgb",
        )
        if st.session_state.get("preset_rgb_path"):
            st.caption(
                f"Imagen demo RGB precargada: `{Path(st.session_state.preset_rgb_path).name}`"
            )
    with u2:
        ir_upload = st.file_uploader(
            "Captura termográfica IR (opcional)",
            type=["jpg", "jpeg", "png", "webp"],
            key="uploader_ir",
        )
        if st.session_state.get("preset_thermal_path"):
            st.caption(
                f"Imagen demo IR precargada: `{Path(st.session_state.preset_thermal_path).name}`"
            )

    q_col, t_col = st.columns([3, 1], vertical_alignment="bottom")
    with q_col:
        user_query = st.text_input(
            "Consulta específica al asistente documental (opcional)",
            key="inp_user_query",
            placeholder="Ej. ¿Qué controles manuales se recomiendan ante baja generación y suciedad visible?",
        )
    with t_col:
        force_visual = st.toggle(
            "Forzar inspección visual",
            key="inp_force_visual",
        )

    submitted = st.form_submit_button(
        "Ejecutar diagnóstico SolarGuard AI",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        key="btn_run_inspection",
    )

# ==============================================================================
# 3. Ejecución del Grafo LangGraph al enviar el formulario
# ==============================================================================
if submitted:
    rgb_path = (
        save_uploaded_file(rgb_upload, prefix="rgb")
        if rgb_upload is not None
        else st.session_state.get("preset_rgb_path")
    )
    ir_path = (
        save_uploaded_file(ir_upload, prefix="ir")
        if ir_upload is not None
        else st.session_state.get("preset_thermal_path")
    )

    graph = get_solarguard_graph()
    thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: Dict[str, Any] = {
        "panel_id": panel_id.strip() or "PLANTA_GENERAL",
        "telemetry": {
            "fecha": fecha.strip(),
            "irradiancia": float(irradiancia),
            "temp_amb": float(temp_amb),
            "temp_mod": float(temp_mod),
            "capacidad_instalada": float(capacidad),
            "energia_observada": float(energia_observada),
        },
        "rgb_image_path": rgb_path,
        "thermal_image_path": ir_path,
        "user_query": user_query.strip() or None,
        "force_visual_inspection": bool(force_visual),
    }

    t0 = time.perf_counter()
    with st.spinner("Ejecutando expertos y enrutamiento LangGraph..."):
        state_out = graph.invoke(initial_state, config=config)
        snapshot = graph.get_state(config)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    st.session_state.active_thread_id = thread_id
    st.session_state.last_agent_state = state_out
    st.session_state.last_latency_ms = elapsed_ms
    st.session_state.hitl_pending = bool(snapshot.next and "validar_tecnico" in snapshot.next)
    logger.info(
        f"[UI] Inspección ejecutada thread={thread_id}, latency_ms={elapsed_ms}, "
        f"hitl_pending={st.session_state.hitl_pending}"
    )

# ==============================================================================
# 4. Presentación de resultados, evidencia visual, HITL y trazas
# ==============================================================================
state_data = st.session_state.get("last_agent_state")
if state_data:
    report = state_data.get("structured_report") or {}
    errors = state_data.get("errors") or []
    uncertainty = bool(state_data.get("uncertainty_flag", False))
    latency_ms = st.session_state.get("last_latency_ms")

    st.divider()
    st.subheader("Resultados del análisis integrado", icon=":material/monitoring:")

    # Fila KPI con atributos homogéneos en todas las tarjetas.
    exp_val = state_data.get("expected_energy_kwh")
    disc_val = state_data.get("energy_discrepancy_pct")
    rgb_cls = state_data.get("rgb_classification") or {}
    rgb_lbl = rgb_cls.get("label", "No requerida")
    rgb_conf = rgb_cls.get("confidence")

    with st.container(horizontal=True):
        st.metric(
            "Energía esperada",
            f"{exp_val:.2f} kWh/h" if exp_val is not None else "N/D",
            border=True,
            icon=":material/bolt:",
        )
        st.metric(
            "Déficit observado",
            f"{disc_val:.2f}%" if disc_val is not None else "N/D",
            border=True,
            icon=":material/monitoring:",
        )
        st.metric(
            "Diagnóstico visual RGB",
            f"{rgb_lbl}" + (f" ({rgb_conf:.2f})" if rgb_conf is not None else ""),
            border=True,
            icon=":material/visibility:",
        )
        st.metric(
            "Estado orden (HITL)",
            str(report.get("estado_orden", "SIN_ORDEN")),
            border=True,
            icon=":material/verified_user:",
        )

    if latency_ms is not None:
        st.caption(f"Tiempo total de inferencia y orquestación: **{latency_ms:.1f} ms**")

    # Alertas principales con parámetro `title=`.
    if errors:
        st.error(
            "\n".join(f"• {e}" for e in errors),
            title="Error de validación en entradas",
            icon=":material/error:",
        )
    elif uncertainty:
        st.warning(
            state_data.get("uncertainty_reason")
            or "La captura visual no es evaluable o tiene confianza menor a 0.60.",
            title="Diagnóstico detenido por incertidumbre",
            icon=":material/warning:",
        )
    elif st.session_state.get("hitl_pending"):
        st.warning(
            "Propuesta técnica generada y validada por guardrails. "
            "Requiere aprobación o rechazo explícito del técnico antes de emitir la orden.",
            title="Intervención humana requerida (Human-in-the-Loop)",
            icon=":material/verified_user:",
        )
    elif state_data.get("order_record"):
        ord_id = state_data["order_record"].get("order_id", "")
        st.success(
            f"Orden simulada **{ord_id}** aprobada por técnico y persistida en `artifacts/orders/revision_orders.json`.",
            title="Orden de revisión emitida",
            icon=":material/check_circle:",
        )
    else:
        st.info(
            f"Discrepancia energética inferior al umbral ({ENERGY_DISCREPANCY_THRESHOLD_PCT:.0f}%). "
            "No se requiere intervención técnica.",
            title="Operación dentro de línea base normal",
            icon=":material/info:",
        )

    sec_warns = report.get("advertencias_seguridad") or []
    if sec_warns:
        st.warning(
            "\n".join(f"• {w}" for w in sec_warns),
            title="Guardrails operativos y seguridad",
            icon=":material/shield:",
        )

    # Panel HITL cuando el grafo está pausado antes de `validar_tecnico`.
    if st.session_state.get("hitl_pending") and st.session_state.get("active_thread_id"):
        with st.container(border=True):
            st.subheader("Validación del técnico (Human-in-the-Loop)", icon=":material/verified_user:")
            reviewer_note = st.text_input(
                "Observación del técnico revisor",
                value="Verificado en tablero SolarGuard AI.",
                key="hitl_note_input",
            )
            b1, b2 = st.columns(2)
            with b1:
                approve_clicked = st.button(
                    "Aprobar y emitir orden simulada",
                    type="primary",
                    icon=":material/verified_user:",
                    width="stretch",
                    key="btn_hitl_approve",
                )
            with b2:
                reject_clicked = st.button(
                    "Rechazar propuesta",
                    type="secondary",
                    icon=":material/cancel:",
                    width="stretch",
                    key="btn_hitl_reject",
                )

            if approve_clicked or reject_clicked:
                graph = get_solarguard_graph()
                cfg = {"configurable": {"thread_id": st.session_state.active_thread_id}}
                graph.update_state(
                    cfg,
                    {
                        "human_approved": bool(approve_clicked),
                        "human_reviewer_note": reviewer_note,
                    },
                )
                resumed_state = graph.invoke(None, config=cfg)
                st.session_state.last_agent_state = resumed_state
                st.session_state.hitl_pending = False
                st.rerun()

    # Columnas con evidencia visual y documental.
    left_col, right_col = st.columns(2)
    with left_col:
        with st.container(border=True):
            st.subheader("Evidencia visual (YOLO + RGB + IR)", icon=":material/visibility:")
            rgb_img_p = state_data.get("rgb_image_path")
            crops = state_data.get("detected_panels_crops") or []
            if rgb_img_p and Path(rgb_img_p).exists():
                st.image(
                    rgb_img_p,
                    caption=f"Captura RGB evaluada · {rgb_lbl}",
                    alt="Fotografía RGB del panel solar evaluado",
                    width="stretch",
                )
            if crops and crops[0] != rgb_img_p and Path(crops[0]).exists():
                st.image(
                    crops[0],
                    caption="Recorte espacial detectado por YOLOv8n",
                    alt="Recorte de panel detectado por YOLO",
                    width="stretch",
                )
            all_probs = rgb_cls.get("all_probs") or {}
            if all_probs:
                prob_df = pd.DataFrame(
                    [{"Clase": k, "Probabilidad": float(v)} for k, v in all_probs.items()]
                )
                st.bar_chart(
                    prob_df,
                    x="Clase",
                    y="Probabilidad",
                    alt="Distribución de probabilidades del clasificador RGB",
                )
            elif not rgb_img_p:
                st.caption("No se adjuntó imagen RGB o no fue requerida por discrepancia < 15%.")

            ir_cls = state_data.get("thermal_classification")
            if ir_cls:
                st.divider()
                st.markdown(
                    f"**Termografía IR (Experto 3):** `{ir_cls['label']}` "
                    f"(confianza `{ir_cls['confidence']:.2f}`)"
                )
                ir_p = state_data.get("thermal_image_path")
                if ir_p and Path(ir_p).exists():
                    st.image(
                        ir_p,
                        caption="Parche termográfico IR analizado",
                        alt="Captura termográfica infrarroja del módulo",
                        width=220,
                    )

    with right_col:
        with st.container(border=True):
            st.subheader("Recomendación y respaldo RAG", icon=":material/menu_book:")
            st.markdown(f"**Panel:** `{report.get('panel', 'N/D')}`")
            st.markdown(f"**Energía esperada:** {report.get('energia_esperada', 'N/D')}")
            st.markdown(f"**Desviación:** {report.get('desviacion', 'N/D')}")

            st.markdown("**Pasos de revisión recomendados:**")
            for idx, step in enumerate(report.get("pasos_revision", []), start=1):
                st.markdown(f"{idx}. {step}")

            st.markdown("**Citas documentales:**")
            citas = report.get("citas") or []
            if citas:
                for cita in citas:
                    st.markdown(f"- `{cita}`")
            else:
                st.caption("Sin respaldo documental aplicable para la consulta realizada.")

            retrieved_docs = state_data.get("retrieved_docs") or []
            if retrieved_docs:
                docs_exp = st.expander(
                    f"Ver fragmentos recuperados ({len(retrieved_docs)})",
                    on_change="rerun",
                )
                if docs_exp.open:
                    with docs_exp:
                        for doc in retrieved_docs:
                            st.caption(
                                f"**{doc['source']} (p. {doc['page']}, score={doc['score']:.3f})**"
                            )
                            st.text(doc["content"])

    # Trazas de ejecución LangGraph y descarga de reporte.
    trace_exp = st.expander("Trazas paso a paso del grafo LangGraph", on_change="rerun")
    if trace_exp.open:
        with trace_exp:
            for trace_line in state_data.get("trace_log", []):
                with st.status(trace_line, type="step", state="complete"):
                    pass

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Descargar reporte Markdown (.md)",
            data=build_markdown_report(state_data),
            file_name=f"reporte_{report.get('panel', 'panel')}.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
            key="btn_dl_md",
        )
    with dl2:
        st.download_button(
            "Descargar reporte estructurado (.json)",
            data=build_json_report(state_data),
            file_name=f"reporte_{report.get('panel', 'panel')}.json",
            mime="application/json",
            icon=":material/download:",
            width="stretch",
            key="btn_dl_json",
        )
