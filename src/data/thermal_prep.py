"""
Preparación balanceada del dataset de Termografía Infrarroja (Experto 3 — Condicional).

Paso a paso:
1. Lee `datasets/InfraredSolarModules/module_metadata.json` (20,000 módulos Raptor Maps).
2. Filtra y agrupa hacia las 3 clases térmicas canónicas:
   - `Normal térmico`          <- `No-Anomaly`
   - `Hotspot / Celda crítica` <- `Hot-Spot`, `Hot-Spot-Multi`, `Cell`, `Cell-Multi`
   - `Fallo de diodo / Bypass` <- `Diode`, `Diode-Multi`
3. Construye una submuestra balanceada de 2,000 imágenes (~667 por clase, semilla fija 42)
   verificando la legibilidad de cada parche térmico en disco.
4. Realiza partición estratificada 70% train / 15% val / 15% test y persiste
   `artifacts/splits/thermal_splits.csv` y `artifacts/splits/thermal_splits_summary.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.core.config import (
    IR_DATASET_DIR,
    RANDOM_SEED,
    SPLITS_DIR,
    THERMAL_CLASSES,
    THERMAL_CLASS_MAPPING,
)
from src.core.logging_config import get_logger
from src.core.telemetry import track_event

logger = get_logger(__name__)


def prepare_thermal_splits(
    dataset_dir: Path = IR_DATASET_DIR,
    total_target: int = 2000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Genera y guarda los splits balanceados (70/15/15) para el clasificador térmico IR.
    """
    with track_event(
        logger,
        "prepare_thermal_splits",
        dataset_dir=str(dataset_dir),
        total_target=total_target,
    ) as ctx:
        meta_path = dataset_dir / "module_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No se encontró {meta_path}")

        raw_meta: Dict[str, Dict[str, str]] = json.loads(meta_path.read_text(encoding="utf-8"))
        rows: List[Dict[str, str]] = []

        # Paso 1: Filtrar únicamente las categorías cubiertas por el mapeo de 3 clases.
        for mod_id, info in raw_meta.items():
            raw_cls = info.get("anomaly_class", "")
            if raw_cls not in THERMAL_CLASS_MAPPING:
                continue
            rel_path = info.get("image_filepath", "")
            abs_path = (dataset_dir / rel_path).resolve()
            if not abs_path.exists():
                continue
            rows.append(
                {
                    "module_id": str(mod_id),
                    "image_path": str(abs_path),
                    "raw_class": raw_cls,
                    "label": THERMAL_CLASS_MAPPING[raw_cls],
                }
            )

        full_df = pd.DataFrame(rows)
        logger.info(
            f"Parches IR elegibles encontrados: {len(full_df)} "
            f"en clases={full_df['label'].value_counts().to_dict()}"
        )

        # Paso 2: Submuestreo balanceado (~2,000 imágenes en total repartidas entre 3 clases).
        base_per_class = total_target // len(THERMAL_CLASSES)
        remainder = total_target % len(THERMAL_CLASSES)

        sampled_parts: List[pd.DataFrame] = []
        for idx, cls_name in enumerate(THERMAL_CLASSES):
            cls_df = full_df[full_df["label"] == cls_name]
            quota = base_per_class + (1 if idx < remainder else 0)
            n_take = min(quota, len(cls_df))
            sampled = cls_df.sample(n=n_take, random_state=seed + idx)
            sampled_parts.append(sampled)

        balanced_df = pd.concat(sampled_parts, ignore_index=True)

        # Verificar integridad de lectura en las muestras seleccionadas.
        valid_mask = []
        for p in balanced_df["image_path"]:
            try:
                with Image.open(p) as im:
                    im.verify()
                valid_mask.append(True)
            except Exception:
                valid_mask.append(False)
        balanced_df = balanced_df[valid_mask].reset_index(drop=True)

        # Paso 3: Partición estratificada 70% train / 15% val / 15% test.
        train_df, temp_df = train_test_split(
            balanced_df,
            test_size=0.30,
            stratify=balanced_df["label"],
            random_state=seed,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            stratify=temp_df["label"],
            random_state=seed,
        )

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        splits_csv = SPLITS_DIR / "thermal_splits.csv"
        final_df.to_csv(splits_csv, index=False)

        summary = {
            "total_sampled": int(len(final_df)),
            "splits": {
                sp: {lbl: int(cnt) for lbl, cnt in sub["label"].value_counts().items()}
                for sp, sub in final_df.groupby("split")
            },
            "canonical_classes": THERMAL_CLASSES,
        }
        summary_path = SPLITS_DIR / "thermal_splits_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        ctx["total_sampled"] = len(final_df)
        logger.info(f"Manifiesto térmico guardado en {splits_csv} ({len(final_df)} muestras).")
        return final_df


if __name__ == "__main__":
    prepare_thermal_splits()
