"""
Entrenamiento y evaluación del Experto 2A: Detector Espacial de Paneles (YOLOv8n).

Paso a paso:
1. Aplica límites estrictos de CPU (`configure_cpu_limits`) para entrenar únicamente en CPU
   sin bloquear la máquina (`workers=0`, `batch=8`, `imgsz=320`, `cache=False`, `amp=False`).
2. Prepara los splits y `configs/yolo_data.yaml` (85% train, 15% val, test intacto).
3. Entrena `yolov8n.pt` durante 6 épocas (ajuste ligero con `fraction=0.6` si se desea o
   dataset completo con `imgsz=320`; 6 épocas a 320px completan rápido y producen cajas sólidas).
4. Evalúa el modelo final sobre el conjunto `test` intacto reportando `mAP@50`, `mAP@50-95`,
   `Precision` y `Recall`.
5. Copia el mejor checkpoint a `artifacts/models/yolo_panel_detector.pt`, exporta métricas
   a `artifacts/metrics/yolo_metrics.json` y genera una figura visual de detecciones reales
   sobre muestras de test en `report/figures/yolo_detections.png`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.core.config import (  # noqa: E402
    ARTIFACTS_DIR,
    FIGURES_DIR,
    MAX_CPU_THREADS,
    METRICS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    SPLITS_DIR,
    configure_cpu_limits,
)
from src.core.logging_config import get_logger, log_section  # noqa: E402
from src.core.telemetry import log_metric, track_event  # noqa: E402
from src.data.yolo_prep import prepare_yolo_dataset  # noqa: E402

logger = get_logger(__name__)


def _find_base_yolo_weights() -> str:
    """Busca `yolov8n.pt` localmente para operar sin acceso a internet."""
    for candidate in (
        PROJECT_ROOT / "yolov8n.pt",
        MODELS_DIR / "yolov8n.pt",
        ARTIFACTS_DIR / "cache" / "yolov8n.pt",
    ):
        if candidate.exists():
            return str(candidate)
    return "yolov8n.pt"


def _render_sample_detections(model: YOLO, test_txt: Path, out_path: Path) -> None:
    """Genera una cuadrícula con hasta 4 imágenes de test anotadas con cajas detectadas."""
    lines = [ln.strip() for ln in test_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sample_paths = lines[:4]
    if not sample_paths:
        return

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
    for ax, img_p in zip(axes.ravel(), sample_paths):
        res = model.predict(source=img_p, imgsz=320, conf=0.25, device="cpu", verbose=False)[0]
        plotted_bgr = res.plot()
        plotted_rgb = plotted_bgr[:, :, ::-1]
        ax.imshow(plotted_rgb)
        ax.set_title(f"{Path(img_p).name[:26]} (cajas={len(res.boxes)})", fontsize=8.5)
        ax.axis("off")

    for ax in axes.ravel()[len(sample_paths) :]:
        ax.axis("off")

    fig.suptitle("Detecciones Espaciales YOLOv8n en Conjunto Test Intacto", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def train_and_evaluate_yolo(
    epochs: int = 6,
    imgsz: int = 320,
    batch: int = 8,
    fraction: float = 0.65,
) -> Dict[str, Any]:
    """
    Entrena YOLOv8n en CPU con restricciones conservadoras de memoria/hilos
    y evalúa `mAP@50` sobre el split `test`.
    """
    configure_cpu_limits(MAX_CPU_THREADS)
    log_section(logger, "ENTRENAMIENTO DETECTOR ESPACIAL YOLO (EXPERTO 2A)")

    prep_info = prepare_yolo_dataset()
    yaml_path = str(prep_info["yaml_path"])
    weights_src = _find_base_yolo_weights()
    runs_project = ARTIFACTS_DIR / "yolo_runs"

    with track_event(
        logger,
        "train_yolov8n_cpu",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        fraction=fraction,
    ) as ctx:
        model = YOLO(weights_src)
        # Entrenamiento liviano en CPU: workers=0 evita multiprocessing pesado en RAM.
        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            fraction=fraction,
            device="cpu",
            workers=0,
            cache=False,
            amp=False,
            patience=4,
            seed=RANDOM_SEED,
            project=str(runs_project),
            name="solarguard_yolov8n",
            exist_ok=True,
            verbose=False,
        )

        best_run_weights = runs_project / "solarguard_yolov8n" / "weights" / "best.pt"
        if not best_run_weights.exists():
            best_run_weights = runs_project / "solarguard_yolov8n" / "weights" / "last.pt"

        final_model_path = MODELS_DIR / "yolo_panel_detector.pt"
        shutil.copy2(best_run_weights, final_model_path)
        logger.info(f"Pesos YOLO copiados a {final_model_path}")

        # Evaluación formal sobre el split `test` intacto.
        best_model = YOLO(str(final_model_path))
        test_metrics = best_model.val(
            data=yaml_path,
            split="test",
            imgsz=imgsz,
            batch=batch,
            device="cpu",
            workers=0,
            verbose=False,
        )

        box_metrics = test_metrics.box
        map50 = round(float(box_metrics.map50), 4)
        map50_95 = round(float(box_metrics.map), 4)
        precision = round(float(box_metrics.mp), 4)
        recall = round(float(box_metrics.mr), 4)

        # Métricas desglosadas por clase si están disponibles.
        per_class: Dict[str, Dict[str, float]] = {}
        class_names: List[str] = list(prep_info["classes"])  # type: ignore[arg-type]
        if hasattr(box_metrics, "ap50") and len(box_metrics.ap50) == len(class_names):
            for idx, cname in enumerate(class_names):
                per_class[cname] = {"ap50": round(float(box_metrics.ap50[idx]), 4)}

        ctx["test_map50"] = map50
        ctx["test_map50_95"] = map50_95

    _render_sample_detections(
        best_model,
        SPLITS_DIR / "yolo_test.txt",
        FIGURES_DIR / "yolo_detections.png",
    )

    metrics_payload = {
        "model_name": "YOLOv8n_Panel_Detector",
        "train_config": {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "fraction": fraction,
            "device": "cpu",
        },
        "splits": {
            "train": prep_info["train_count"],
            "val": prep_info["val_count"],
            "test": prep_info["test_count"],
        },
        "test_metrics": {
            "mAP50": map50,
            "mAP50_95": map50_95,
            "precision": precision,
            "recall": recall,
        },
        "per_class_ap50": per_class,
        "model_artifact": str(final_model_path),
    }

    metrics_path = METRICS_DIR / "yolo_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_metric(logger, "vision.yolo_detector", metrics_payload["test_metrics"])
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate_yolo()
