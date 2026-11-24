"""
Auditoría de Órdenes Simuladas (HITL) y Visor del Registro Diario `.log`.

Permite verificar en tiempo real:
1. Las órdenes técnicas emitidas tras aprobación humana (`artifacts/orders/revision_orders.json`).
2. El archivo de eventos diarios (`logs/solarguard_YYYY-MM-DD.log`) filtrado por nivel de severidad.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.ui.helpers import load_daily_log_lines, load_orders

# ==============================================================================
# 1. Registro de Órdenes Simuladas (HITL)
# ==============================================================================
st.subheader("Órdenes de revisión técnica simuladas", icon=":material/verified_user:")
orders = load_orders()

if orders:
    summary_rows = [
        {
            "ID Orden": o.get("order_id"),
            "Fecha UTC": o.get("created_at_utc"),
            "Panel": o.get("panel_id"),
            "Prioridad": o.get("priority"),
            "Diagnóstico": o.get("diagnosis_summary"),
            "Aprobado HITL": o.get("approved_by_human"),
            "Nota": o.get("reviewer_note"),
        }
        for o in orders
    ]
    st.dataframe(
        pd.DataFrame(summary_rows),
        hide_index=True,
        alt="Listado de órdenes de revisión técnica simuladas",
    )
    st.download_button(
        "Descargar registro de órdenes (.json)",
        data=json.dumps(orders, indent=2, ensure_ascii=False),
        file_name="revision_orders.json",
        mime="application/json",
        icon=":material/download:",
        key="btn_dl_orders",
    )
else:
    st.info(
        "Aún no se han emitido órdenes simuladas. Aprueba una propuesta en **Inspección integral**.",
        title="Sin órdenes registradas",
        icon=":material/info:",
    )

st.divider()

# ==============================================================================
# 2. Visor en vivo del archivo `.log` diario
# ==============================================================================
st.subheader("Registro diario de eventos y telemetría", icon=":material/receipt_long:")

f_col, b_col = st.columns([3, 1], vertical_alignment="bottom")
with f_col:
    selected_levels = st.pills(
        "Filtrar por nivel",
        options=["INFO", "WARNING", "ERROR", "DEBUG"],
        selection_mode="multi",
        default=["INFO", "WARNING", "ERROR"],
        key="log_level_pills",
    )
with b_col:
    st.button(
        "Actualizar registros",
        icon=":material/refresh:",
        width="stretch",
        key="btn_refresh_logs",
    )

log_path, lines = load_daily_log_lines(max_lines=250)
st.caption(f"Archivo diario activo: `{log_path}` ({len(lines)} líneas recientes)")

if selected_levels:
    filtered = [
        ln for ln in lines if any(f"| {lvl:<8}" in ln for lvl in selected_levels)
    ]
else:
    filtered = lines

if filtered:
    st.code("\n".join(reversed(filtered[-100:])), language="log")
else:
    st.caption("No hay registros que coincidan con los niveles seleccionados.")
