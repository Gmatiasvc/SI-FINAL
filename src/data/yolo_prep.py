"""
Preparación de listas de entrenamiento/validación/prueba para el Detector Espacial YOLO (Experto 2A).

Paso a paso:
1. Audita las imágenes y etiquetas YOLO reales en `datasets/mixed_1 Computer Vision Dataset/`.
   Nota: El dataset original contiene `train/` (1,287 imágenes) y `test/` (228 imágenes), sin
   carpeta `valid/` separada.
2. Para mantener `test/` completamente intacto y libre de fugas, divide `train/` de forma
   reproducible (85% entrenamiento, 15% validación) verificando que cada imagen posea su
   archivo `.txt` de anotación correspondiente.
3. Escribe listas absolutas en `artifacts/splits/yolo_{train,val,test}.txt` y genera
   `configs/yolo_data.yaml` apuntando a dichas listas con las 3 clases (`double_panel`,
   `dummy`, `single_panel`).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import yaml

from src.core.config import CONFIGS_DIR, RANDOM_SEED, SPLITS_DIR, YOLO_DATASET_DIR
from src.core.logging_config import get_logger
from src.core.telemetry import track_event

logger = get_logger(__name__)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _collect_labeled_images(images_dir: Path, labels_dir: Path) -> List[Path]:
    """Recopila imágenes que cuentan con su archivo de anotación `.txt` en `labels_dir`."""
    valid_paths: List[Path] = []
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTS:
            continue
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            valid_paths.append(img_path.resolve())
    return valid_paths


def prepare_yolo_dataset(
    dataset_dir: Path = YOLO_DATASET_DIR,
    val_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Dict[str, object]:
    """
    Genera los archivos `.txt` de splits y `configs/yolo_data.yaml` para entrenar YOLOv8n.
    """
    with track_event(logger, "prepare_yolo_dataset", dataset_dir=str(dataset_dir)) as ctx:
        src_yaml = dataset_dir / "data.yaml"
        if not src_yaml.exists():
            raise FileNotFoundError(f"No se encontró {src_yaml}")

        raw_cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
        class_names = raw_cfg.get("names", ["double_panel", "dummy", "single_panel"])

        train_pool = _collect_labeled_images(
            dataset_dir / "train" / "images",
            dataset_dir / "train" / "labels",
        )
        test_pool = _collect_labeled_images(
            dataset_dir / "test" / "images",
            dataset_dir / "test" / "labels",
        )

        rng = random.Random(seed)
        shuffled = list(train_pool)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(val_ratio * len(shuffled))))
        val_list = shuffled[:n_val]
        train_list = shuffled[n_val:]

        train_txt = SPLITS_DIR / "yolo_train.txt"
        val_txt = SPLITS_DIR / "yolo_val.txt"
        test_txt = SPLITS_DIR / "yolo_test.txt"

        train_txt.write_text("\n".join(str(p) for p in train_list) + "\n", encoding="utf-8")
        val_txt.write_text("\n".join(str(p) for p in val_list) + "\n", encoding="utf-8")
        test_txt.write_text("\n".join(str(p) for p in test_pool) + "\n", encoding="utf-8")

        yolo_cfg = {
            "path": str(dataset_dir.resolve()),
            "train": str(train_txt.resolve()),
            "val": str(val_txt.resolve()),
            "test": str(test_txt.resolve()),
            "nc": len(class_names),
            "names": class_names,
        }
        yaml_out = CONFIGS_DIR / "yolo_data.yaml"
        yaml_out.write_text(yaml.safe_dump(yolo_cfg, sort_keys=False), encoding="utf-8")

        summary = {
            "yaml_path": str(yaml_out.resolve()),
            "train_count": len(train_list),
            "val_count": len(val_list),
            "test_count": len(test_pool),
            "classes": class_names,
        }
        (SPLITS_DIR / "yolo_splits_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ctx.update(
            train_count=len(train_list),
            val_count=len(val_list),
            test_count=len(test_pool),
        )
        logger.info(
            f"Configuración YOLO generada en {yaml_out} "
            f"(train={len(train_list)}, val={len(val_list)}, test={len(test_pool)})."
        )
        return summary


if __name__ == "__main__":
    prepare_yolo_dataset()
