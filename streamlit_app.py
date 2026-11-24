"""
Punto de entrada principal (multipágina) de la interfaz web SolarGuard AI en Streamlit.

Paso a paso:
1. Configura la página (`st.set_page_config`) con ícono Material y layout ancho.
2. Inicializa las llaves globales de `st.session_state` de forma segura una sola vez.
3. Define la navegación superior moderna mediante `st.navigation(..., position="top")` y
   `st.Page` apuntando a `app_pages/` (Inspección integral, Asistente RAG, Métricas y Trazas).
4. Muestra en la barra lateral el estado en vivo de los 4 modelos entrenados y el índice FAISS.
5. Renderiza el título principal de la página activa una única vez antes de ejecutarla.
"""

from __future__ import annotations

import streamlit as st

from src.ui.helpers import get_system_health_status, load_demo_presets

st.set_page_config(
    page_title="SolarGuard AI · Inspección Fotovoltaica",
    page_icon=":material/solar_power:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paso 1: Inicialización de estado global compartido entre páginas.
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None
if "last_agent_state" not in st.session_state:
    st.session_state.last_agent_state = None
if "hitl_pending" not in st.session_state:
    st.session_state.hitl_pending = False
if "last_latency_ms" not in st.session_state:
    st.session_state.last_latency_ms = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "demo_preset_control" not in st.session_state:
    st.session_state.demo_preset_control = "Manual"

# Inicializar valores del formulario de inspección si aún no existen en sesión.
_defaults = load_demo_presets()["Manual"]
_field_defaults = {
    "inp_panel_id": _defaults["panel_id"],
    "inp_fecha": _defaults["fecha"],
    "inp_irradiancia": float(_defaults["irradiancia"]),
    "inp_temp_amb": float(_defaults["temp_amb"]),
    "inp_temp_mod": float(_defaults["temp_mod"]),
    "inp_capacidad": float(_defaults["capacidad"]),
    "inp_energia_observada": float(_defaults["energia_observada"]),
    "inp_user_query": _defaults["user_query"],
    "inp_force_visual": bool(_defaults["force_visual"]),
    "preset_rgb_path": _defaults["rgb_image_path"],
    "preset_thermal_path": _defaults["thermal_image_path"],
}
for _key, _val in _field_defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# Paso 2: Registro de páginas en barra de navegación superior.
pages = [
    st.Page(
        "app_pages/inspeccion.py",
        title="Inspección integral",
        icon=":material/solar_power:",
        url_path="inspeccion",
        default=True,
        help=(
            "Ejecuta el grafo MoE completo: predicción tabular, visión YOLO+RGB, "
            "termografía IR opcional, citas RAG y validación humana."
        ),
    ),
    st.Page(
        "app_pages/asistente.py",
        title="Asistente RAG",
        icon=":material/chat:",
        url_path="asistente",
        help=(
            "Chatbot técnico contextual conectado al corpus PDF y guardrails "
            "para responder dudas sobre el diagnóstico y manuales."
        ),
    ),
    st.Page(
        "app_pages/metricas.py",
        title="Métricas y evaluación",
        icon=":material/monitoring:",
        url_path="metricas",
        help=(
            "Tablero comparativo de resultados reales en test para ML tabular, "
            "clasificador RGB, IR, YOLOv8n y benchmark RAG."
        ),
    ),
    st.Page(
        "app_pages/trazas_logs.py",
        title="Trazas y órdenes",
        icon=":material/receipt_long:",
        url_path="trazas",
        help=(
            "Auditoría de órdenes de revisión simuladas, trazas de nodos LangGraph "
            "y visor del archivo de log diario."
        ),
    ),
]

current_page = st.navigation(pages, position="top")

# Paso 3: Barra lateral compacta con estado de artefactos y políticas operativas.
with st.sidebar:
    st.subheader("Estado del sistema", icon=":material/memory:")
    health_items = get_system_health_status()
    for item in health_items:
        badge = ":green[:material/check_circle:]" if item["ready"] else ":red[:material/error:]"
        st.caption(f"{badge} {item['component']} ({item['size_mb']:.1f} MB)")

    st.divider()
    st.subheader("Políticas de seguridad", icon=":material/shield:")
    st.caption(
        "• Prohibida desconexión eléctrica automatizada.\n"
        "• Parada automática si confianza RGB < 0.60.\n"
        "• Aprobación humana obligatoria (HITL) antes de emitir orden."
    )

# Paso 4: Título único de página manejado desde el entrypoint.
st.title(current_page.title, icon=current_page.icon)
current_page.run()
