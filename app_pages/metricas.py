"""
Tablero de Métricas Experimentales y Evaluación Comparativa de todos los Expertos.

Lee directamente los archivos verificables en `artifacts/metrics/*.json` y las figuras
generadas en `report/figures/*.png` usando `@st.cache_data`.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.helpers import get_figure_path, load_json_metrics

tab_m = load_json_metrics("tabular_metrics.json")
rgb_m = load_json_metrics("rgb_metrics.json")
ir_m = load_json_metrics("thermal_metrics.json")
yolo_m = load_json_metrics("yolo_metrics.json")
rag_m = load_json_metrics("rag_benchmark.json")

# ==============================================================================
# 1. Fila KPI con los resultados reales en conjunto de prueba intacto
# ==============================================================================
best_tab_name = tab_m.get("best_model_selected_by_val_rmse", "GradientBoostingRegressor")
best_tab_test = (tab_m.get("models", {}).get(best_tab_name, {}) or {}).get("test", {})

with st.container(horizontal=True):
    st.metric(
        "ML Tabular · nMAE Test",
        f"{best_tab_test.get('nmae_pct', 0.0):.3f}%",
        border=True,
        icon=":material/bolt:",
    )
    st.metric(
        "Visión RGB · Macro-F1 Test",
        f"{rgb_m.get('finetuned', {}).get('test_macro_f1', 0.0):.4f}",
        border=True,
        icon=":material/visibility:",
    )
    st.metric(
        "Termografía IR · Macro-F1 Test",
        f"{ir_m.get('test_macro_f1', 0.0):.4f}",
        border=True,
        icon=":material/thermostat:",
    )
    st.metric(
        "Detector YOLO · mAP@50 Test",
        f"{yolo_m.get('test_metrics', {}).get('mAP50', 0.0):.4f}",
        border=True,
        icon=":material/solar_power:",
    )
    st.metric(
        "RAG · Fidelidad 20Q",
        f"{rag_m.get('comparison', {}).get('con_rag_y_guardrails', {}).get('overall_fidelity_20q', 0.0) * 100:.0f}%",
        border=True,
        icon=":material/menu_book:",
    )

st.divider()

# Selector segmentado (evita renderizar/calcular vistas inactivas como en st.tabs sin control).
view = st.segmented_control(
    "Seleccionar módulo para inspección detallada",
    options=["Tabular ML", "Visión RGB", "Termografía IR", "Detector YOLO", "Benchmark RAG"],
    default="Tabular ML",
    key="metrics_view_selector",
)

if view == "Tabular ML" and tab_m:
    st.subheader("Experto 1 · Regresión temporal de `energia_kwh_hora`")
    rows = []
    for m_name, m_data in tab_m.get("models", {}).items():
        for split in ("train", "val", "test"):
            sp = m_data.get(split, {})
            rows.append(
                {
                    "Modelo": m_name,
                    "Split": split,
                    "MAE (kWh)": sp.get("mae_kwh"),
                    "RMSE (kWh)": sp.get("rmse_kwh"),
                    "nMAE (%)": sp.get("nmae_pct"),
                    "nRMSE (%)": sp.get("nrmse_pct"),
                }
            )
    df_tab = pd.DataFrame(rows)
    st.dataframe(df_tab, hide_index=True, alt="Métricas tabulares en train, val y test")

    test_only = df_tab[df_tab["Split"] == "test"]
    st.bar_chart(
        test_only,
        x="Modelo",
        y=["MAE (kWh)", "RMSE (kWh)"],
        alt="Comparación MAE y RMSE en test tabular",
    )

    c1, c2 = st.columns(2)
    with c1:
        p = get_figure_path("tabular_pred_vs_real.png")
        if p:
            st.image(str(p), alt="Dispersión predicción versus energía real", width="stretch")
    with c2:
        p = get_figure_path("tabular_residuals_by_hour.png")
        if p:
            st.image(str(p), alt="Perfil horario de residuales en test", width="stretch")

elif view == "Visión RGB" and rgb_m:
    st.subheader("Experto 2B · Clasificador Visual RGB (4 clases)")
    rep = rgb_m.get("finetuned", {}).get("per_class", {})
    cls_rows = []
    for cls_name in ["panel limpio", "suciedad visible", "grieta visible", "vista no evaluable"]:
        c_data = rep.get(cls_name, {})
        cls_rows.append(
            {
                "Clase": cls_name,
                "Precision": round(float(c_data.get("precision", 0.0)), 4),
                "Recall": round(float(c_data.get("recall", 0.0)), 4),
                "F1-Score": round(float(c_data.get("f1-score", 0.0)), 4),
                "Soporte": int(c_data.get("support", 0)),
            }
        )
    st.dataframe(
        pd.DataFrame(cls_rows),
        hide_index=True,
        alt="Métricas por clase del clasificador visual RGB",
    )
    c1, c2 = st.columns(2)
    with c1:
        p = get_figure_path("rgb_confusion_matrix.png")
        if p:
            st.image(str(p), alt="Matriz de confusión RGB en test", width="stretch")
    with c2:
        p = get_figure_path("rgb_training_curves.png")
        if p:
            st.image(str(p), alt="Curvas de entrenamiento RGB", width="stretch")
    p_err = get_figure_path("rgb_error_gallery.png")
    if p_err:
        st.image(str(p_err), alt="Galería de errores en test RGB", width="stretch")

elif view == "Termografía IR" and ir_m:
    st.subheader("Experto 3 · Clasificador Térmico Condicional IR (3 clases)")
    rep = ir_m.get("per_class", {})
    cls_rows = []
    for cls_name in ["Normal térmico", "Hotspot / Celda crítica", "Fallo de diodo / Bypass"]:
        c_data = rep.get(cls_name, {})
        cls_rows.append(
            {
                "Clase IR": cls_name,
                "Precision": round(float(c_data.get("precision", 0.0)), 4),
                "Recall": round(float(c_data.get("recall", 0.0)), 4),
                "F1-Score": round(float(c_data.get("f1-score", 0.0)), 4),
                "Soporte": int(c_data.get("support", 0)),
            }
        )
    st.dataframe(
        pd.DataFrame(cls_rows),
        hide_index=True,
        alt="Métricas por clase del clasificador térmico IR",
    )
    c1, c2 = st.columns(2)
    with c1:
        p = get_figure_path("thermal_confusion_matrix.png")
        if p:
            st.image(str(p), alt="Matriz de confusión termografía IR", width="stretch")
    with c2:
        p = get_figure_path("thermal_training_curves.png")
        if p:
            st.image(str(p), alt="Curvas de entrenamiento termografía IR", width="stretch")

elif view == "Detector YOLO" and yolo_m:
    st.subheader("Experto 2A · Detector Espacial YOLOv8n")
    tm = yolo_m.get("test_metrics", {})
    pc = yolo_m.get("per_class_ap50", {})
    yolo_rows = [
        {"Métrica": "Global mAP@50", "Valor": tm.get("mAP50")},
        {"Métrica": "Global mAP@50-95", "Valor": tm.get("mAP50_95")},
        {"Métrica": "Precision global", "Valor": tm.get("precision")},
        {"Métrica": "Recall global", "Valor": tm.get("recall")},
    ] + [{"Métrica": f"AP@50 ({k})", "Valor": v.get("ap50")} for k, v in pc.items()]
    st.dataframe(
        pd.DataFrame(yolo_rows),
        hide_index=True,
        alt="Métricas del detector YOLOv8n en test",
    )
    p = get_figure_path("yolo_detections.png")
    if p:
        st.image(str(p), alt="Detecciones espaciales YOLOv8n en test", width="stretch")

elif view == "Benchmark RAG" and rag_m:
    st.subheader("Experto 4 · Benchmark RAG + Guardrails (20 Consultas)")
    comp = rag_m.get("comparison", {}).get("con_rag_y_guardrails", {})
    comp_df = pd.DataFrame(
        [
            {"Indicador": "Recall@3 (10 contestables)", "Con RAG (%)": comp.get("recall_at_3", 0) * 100},
            {"Indicador": "Exactitud página (10)", "Con RAG (%)": comp.get("citation_accuracy_page", 0) * 100},
            {"Indicador": "Abstención sin respaldo (5)", "Con RAG (%)": comp.get("no_support_abstention_rate", 0) * 100},
            {"Indicador": "Bloqueo adversarial (5)", "Con RAG (%)": comp.get("adversarial_rejection_rate", 0) * 100},
            {"Indicador": "Fidelidad global (20Q)", "Con RAG (%)": comp.get("overall_fidelity_20q", 0) * 100},
        ]
    )
    st.bar_chart(
        comp_df,
        x="Indicador",
        y="Con RAG (%)",
        alt="Resultados porcentuales del benchmark RAG de 20 preguntas",
    )
    details_df = pd.DataFrame(rag_m.get("details", []))
    if not details_df.empty:
        st.dataframe(
            details_df[["id", "category", "has_evidence", "top_score", "citations", "query"]],
            hide_index=True,
            alt="Detalle de las 20 preguntas evaluadas en RAG",
        )
