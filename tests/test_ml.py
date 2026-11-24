"""Pruebas automatizadas del módulo Tabular ML (anti-fuga, validación e inferencia)."""

from __future__ import annotations

import json

import pytest

from src.core.config import METRICS_DIR, SPLITS_DIR
from src.ml.predict_tabular import TabularInputError, predecir_generacion


def test_tabular_splits_no_temporal_leakage() -> None:
    info = json.loads((SPLITS_DIR / "tabular_splits_info.json").read_text(encoding="utf-8"))
    assert info["splits"]["train"]["end"] < info["splits"]["val"]["start"]
    assert info["splits"]["val"]["end"] < info["splits"]["test"]["start"]
    assert info["splits"]["train"]["rows"] > 0
    assert info["splits"]["test"]["rows"] > 0


def test_tabular_metrics_and_winner_artifact() -> None:
    metrics = json.loads((METRICS_DIR / "tabular_metrics.json").read_text(encoding="utf-8"))
    assert "LinearRegression_Baseline" in metrics["models"]
    assert "RandomForestRegressor" in metrics["models"]
    assert "GradientBoostingRegressor" in metrics["models"]
    best = metrics["best_model_selected_by_val_rmse"]
    assert metrics["models"][best]["test"]["mae_kwh"] < metrics["models"]["LinearRegression_Baseline"]["test"]["mae_kwh"]


def test_predecir_generacion_normal_and_anomaly() -> None:
    normal = predecir_generacion(
        fecha="2020-06-15T12:00:00",
        condiciones={"irradiancia": 0.75, "temp_amb": 28.0, "temp_mod": 45.0},
        energia_observada=950.0,
    )
    assert normal["expected_energy_kwh"] > 0
    assert normal["uncertainty_kwh"] > 0
    assert normal["is_anomaly"] is False

    anomaly = predecir_generacion(
        fecha="2020-06-15T12:00:00",
        condiciones={"irradiancia": 0.75, "temp_amb": 28.0, "temp_mod": 45.0},
        energia_observada=400.0,
    )
    assert anomaly["is_anomaly"] is True
    assert anomaly["energy_discrepancy_pct"] >= 15.0


def test_predecir_generacion_rejects_invalid_physical_inputs() -> None:
    with pytest.raises(TabularInputError):
        predecir_generacion(
            fecha="2020-06-15T12:00:00",
            condiciones={"irradiancia": -0.2, "temp_amb": 25.0, "temp_mod": 35.0},
        )
