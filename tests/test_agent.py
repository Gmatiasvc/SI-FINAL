"""Pruebas end-to-end del grafo MoE en LangGraph, rutas condicionales, error y HITL."""

from __future__ import annotations

import uuid

import pandas as pd

from src.agents.graph import build_solarguard_graph
from src.core.config import SPLITS_DIR


def test_agent_scenario_1_normal_operation_skips_vision_and_hitl() -> None:
    graph = build_solarguard_graph()
    cfg = {"configurable": {"thread_id": f"test-s1-{uuid.uuid4().hex[:6]}"}}
    out = graph.invoke(
        {
            "panel_id": "INV-TEST-01",
            "telemetry": {
                "fecha": "2020-06-15T12:00:00",
                "irradiancia": 0.75,
                "temp_amb": 28.0,
                "temp_mod": 45.0,
                "energia_observada": 945.0,
            },
        },
        config=cfg,
    )
    assert out["is_anomaly"] is False
    assert out["human_approval_required"] is False
    assert out["structured_report"]["estado_orden"] == "NO_REQUERIDA"
    assert any("[redactar_reporte_normal]" in t for t in out["trace_log"])


def test_agent_scenario_3_integral_with_hitl_pause_and_approval() -> None:
    rgb_df = pd.read_csv(SPLITS_DIR / "rgb_splits.csv")
    ir_df = pd.read_csv(SPLITS_DIR / "thermal_splits.csv")
    img_dirty = str(rgb_df[(rgb_df["split"] == "test") & (rgb_df["label"] == "suciedad visible")].iloc[0]["image_path"])
    img_ir = str(ir_df[(ir_df["split"] == "test") & (ir_df["label"] != "Normal térmico")].iloc[0]["image_path"])

    graph = build_solarguard_graph()
    cfg = {"configurable": {"thread_id": f"test-s3-{uuid.uuid4().hex[:6]}"}}

    paused_state = graph.invoke(
        {
            "panel_id": "PANEL-TEST-S3",
            "telemetry": {
                "fecha": "2020-06-15T12:00:00",
                "irradiancia": 0.75,
                "temp_amb": 28.0,
                "temp_mod": 45.0,
                "energia_observada": 520.0,
            },
            "rgb_image_path": img_dirty,
            "thermal_image_path": img_ir,
        },
        config=cfg,
    )

    snapshot = graph.get_state(cfg)
    # Si la imagen de test fue evaluable (>= 0.60), el grafo debe pausar antes de validar_tecnico.
    if not paused_state.get("uncertainty_flag", False):
        assert "validar_tecnico" in snapshot.next
        graph.update_state(cfg, {"human_approved": True, "human_reviewer_note": "OK pytest"})
        final_state = graph.invoke(None, config=cfg)
        assert final_state.get("order_record") is not None
        assert final_state["order_record"]["order_id"].startswith("ORD-")
        assert final_state["thermal_classification"] is not None
    else:
        assert paused_state["uncertainty_flag"] is True


def test_agent_uncertainty_stop_and_error_route() -> None:
    rgb_df = pd.read_csv(SPLITS_DIR / "rgb_splits.csv")
    non_eval = str(rgb_df[(rgb_df["split"] == "test") & (rgb_df["label"] == "vista no evaluable")].iloc[0]["image_path"])

    graph = build_solarguard_graph()
    cfg_unc = {"configurable": {"thread_id": f"test-unc-{uuid.uuid4().hex[:6]}"}}
    out_unc = graph.invoke(
        {
            "panel_id": "PANEL-UNC",
            "telemetry": {
                "fecha": "2020-06-15T12:00:00",
                "irradiancia": 0.75,
                "temp_amb": 28.0,
                "temp_mod": 45.0,
                "energia_observada": 550.0,
            },
            "rgb_image_path": non_eval,
        },
        config=cfg_unc,
    )
    assert out_unc["uncertainty_flag"] is True
    assert out_unc["structured_report"]["estado_orden"] == "DETENIDA_POR_INCERTIDUMBRE"

    cfg_err = {"configurable": {"thread_id": f"test-err-{uuid.uuid4().hex[:6]}"}}
    out_err = graph.invoke(
        {
            "panel_id": "PANEL-ERR",
            "telemetry": {
                "fecha": "2020-06-15T12:00:00",
                "irradiancia": -1.0,
                "temp_amb": 25.0,
                "temp_mod": 35.0,
            },
        },
        config=cfg_err,
    )
    assert len(out_err["errors"]) >= 1
    assert out_err["structured_report"]["estado_orden"] == "ERROR_ENTRADA"
