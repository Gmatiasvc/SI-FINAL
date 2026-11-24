"""Pruebas automatizadas del módulo de Visión Computacional (YOLO + RGB + Termografía IR)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.config import RGB_CLASSES, SPLITS_DIR, THERMAL_CLASSES
from src.vision.inference import (
    VisionInputError,
    clasificar_panel,
    clasificar_termografia,
)


def test_rgb_splits_have_zero_cluster_leakage() -> None:
    df = pd.read_csv(SPLITS_DIR / "rgb_splits.csv")
    real_df = df[~df["is_synthetic"]]
    assert (real_df.groupby("cluster_id")["split"].nunique() == 1).all()
    assert set(df["label"].unique()) == set(RGB_CLASSES)


def test_clasificar_panel_valid_and_non_evaluable() -> None:
    df = pd.read_csv(SPLITS_DIR / "rgb_splits.csv")
    clean_sample = df[(df["split"] == "test") & (df["label"] == "panel limpio")].iloc[0]["image_path"]
    non_eval_sample = df[(df["split"] == "test") & (df["label"] == "vista no evaluable")].iloc[0]["image_path"]

    res_clean = clasificar_panel(clean_sample, run_yolo=True)
    assert res_clean["label"] in RGB_CLASSES
    assert 0.0 <= res_clean["confidence"] <= 1.0
    assert len(res_clean["detected_panels_crops"]) >= 1

    res_non_eval = clasificar_panel(non_eval_sample, run_yolo=False)
    assert res_non_eval["label"] in RGB_CLASSES
    # Si la etiqueta es vista no evaluable o confianza < 0.60, es_evaluable debe ser False.
    if res_non_eval["label"] == "vista no evaluable":
        assert res_non_eval["es_evaluable"] is False


def test_clasificar_termografia_ir() -> None:
    df = pd.read_csv(SPLITS_DIR / "thermal_splits.csv")
    ir_sample = df[df["split"] == "test"].iloc[0]["image_path"]
    res = clasificar_termografia(ir_sample)
    assert res["label"] in THERMAL_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0


def test_vision_rejects_missing_file() -> None:
    with pytest.raises(VisionInputError):
        clasificar_panel("/tmp/archivo_inexistente_solarguard.jpg")
