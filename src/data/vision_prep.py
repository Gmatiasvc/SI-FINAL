"""
Preparación y deduplicación perceptual anti-fuga del dataset RGB (4 clases obligatorias).

Paso a paso:
1. Escanea recursivamente `datasets/Solar Panel Images Clean and Faulty Images/`,
   filtrando estrictamente archivos de imagen válidos (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`)
   para ignorar archivos del sistema como `desktop.ini`.
2. Aplica el mapeo obligatorio hacia las 4 clases canónicas:
   - `Clean`           -> `panel limpio`
   - `Dusty`, `Bird-drop` -> `suciedad visible`
   - `Physical-Damage` -> `grieta visible`
   - `Snow-Covered`    -> `vista no evaluable`
   Excluye `Electrical-damage` conforme a la restricción ética/técnica de no inferir
   fallos eléctricos internos solo a partir de fotografías RGB.
3. Calcula el hash perceptual (`imagehash.phash`) de cada imagen y agrupa duplicados o tomas
   casi idénticas (distancia de Hamming <= 6) mediante componentes conexos (Union-Find)
   para evitar que versiones aumentadas o tomas consecutivas aparezcan en splits distintos.
4. Divide los clústeres de manera estratificada (70% train / 15% val / 15% test).
5. Genera de forma controlada un subconjunto sintético de `vista no evaluable` (desenfoque
   gaussiano intenso, sobreexposición/glare y oclusión extrema) **dentro de cada split por
   separado**, garantizando cero fuga de datos entre entrenamiento y prueba.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from src.core.config import (
    CACHE_DIR,
    RANDOM_SEED,
    RGB_CLASSES,
    RGB_DATASET_DIR,
    RGB_FOLDER_MAPPING,
    SPLITS_DIR,
)
from src.core.logging_config import get_logger
from src.core.telemetry import track_event

logger = get_logger(__name__)

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class _UnionFind:
    """Estructura Disjoint-Set simple para agrupar imágenes perceptualmente similares."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _create_synthetic_degradation(
    src_path: Path,
    out_path: Path,
    mode: str,
    rng: random.Random,
) -> None:
    """
    Genera una imagen sintética controlada de `vista no evaluable` a partir de una
    imagen perteneciente al mismo split (cero fuga).

    Modos soportados:
    - `blur`: desenfoque gaussiano severo que impide inspeccionar celdas/suciedad.
    - `glare`: sobreexposición extrema con reflejo solar especular blanco.
    - `occlusion`: oclusión severa (>65% del encuadre bloqueado por sombra/objeto).
    """
    with Image.open(src_path) as img:
        rgb = img.convert("RGB").resize((256, 256))
        if mode == "blur":
            degraded = rgb.filter(ImageFilter.GaussianBlur(radius=10))
        elif mode == "glare":
            bright = ImageEnhance.Brightness(rgb).enhance(2.5)
            low_contrast = ImageEnhance.Contrast(bright).enhance(0.45)
            overlay = Image.new("RGB", rgb.size, (255, 255, 250))
            degraded = Image.blend(low_contrast, overlay, alpha=0.55)
        elif mode == "occlusion":
            degraded = rgb.copy()
            draw = ImageDraw.Draw(degraded)
            # Cubrir ~70% de la imagen simulando obstrucción total de cámara o sombra densa.
            if rng.random() < 0.5:
                draw.rectangle([0, 0, 210, 220], fill=(12, 12, 15))
            else:
                draw.rectangle([45, 35, 256, 256], fill=(18, 18, 22))
        else:
            raise ValueError(f"Modo sintético desconocido: {mode}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        degraded.save(out_path, format="JPEG", quality=90)


def prepare_rgb_splits(
    dataset_dir: Path = RGB_DATASET_DIR,
    hamming_threshold: int = 6,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Construye y persiste el manifiesto deduplicado `artifacts/splits/rgb_splits.csv`
    junto con el resumen estadístico `artifacts/splits/rgb_splits_summary.json`.
    """
    with track_event(
        logger,
        "prepare_rgb_splits",
        dataset_dir=str(dataset_dir),
        hamming_threshold=hamming_threshold,
    ) as ctx:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"No existe el directorio RGB: {dataset_dir}")

        records: List[Dict[str, object]] = []
        excluded_counts: Counter[str] = Counter()
        corrupted_files: List[str] = []

        # Paso 1: Descubrimiento de imágenes reales y mapeo a las 4 clases obligatorias.
        for folder in sorted(dataset_dir.iterdir()):
            if not folder.is_dir():
                continue
            raw_class = folder.name
            if raw_class not in RGB_FOLDER_MAPPING:
                n_skipped = sum(
                    1
                    for p in folder.rglob("*")
                    if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
                )
                excluded_counts[raw_class] += n_skipped
                logger.warning(
                    f"Carpeta '{raw_class}' ({n_skipped} imágenes) excluida del modelo RGB "
                    "por restricción ética/técnica (no inferir fallos eléctricos sin medición)."
                )
                continue

            canonical_label = RGB_FOLDER_MAPPING[raw_class]
            for img_path in sorted(folder.rglob("*")):
                if not img_path.is_file() or img_path.suffix.lower() not in VALID_IMAGE_EXTS:
                    continue
                try:
                    with Image.open(img_path) as im:
                        rgb_im = im.convert("RGB")
                        phash = imagehash.phash(rgb_im)
                        width, height = rgb_im.size
                except Exception as exc:
                    corrupted_files.append(f"{img_path}: {exc}")
                    continue

                records.append(
                    {
                        "image_path": str(img_path.resolve()),
                        "raw_folder": raw_class,
                        "label": canonical_label,
                        "phash_hex": str(phash),
                        "phash_bits": phash.hash.flatten().astype(np.uint8),
                        "width": width,
                        "height": height,
                        "is_synthetic": False,
                    }
                )

        n_imgs = len(records)
        logger.info(f"Imágenes RGB válidas indexadas: {n_imgs} (corruptas={len(corrupted_files)})")

        # Paso 2: Agrupación anti-fuga por distancia de Hamming en pHash.
        bit_matrix = np.stack([r["phash_bits"] for r in records])  # (N, 64)
        uf = _UnionFind(n_imgs)
        duplicate_edges = 0
        for i in range(n_imgs):
            # Distancia de Hamming vectorizada contra los elementos posteriores j > i.
            if i + 1 < n_imgs:
                dists = np.sum(bit_matrix[i + 1 :] != bit_matrix[i], axis=1)
                matches = np.where(dists <= hamming_threshold)[0]
                for offset in matches:
                    j = i + 1 + int(offset)
                    uf.union(i, j)
                    duplicate_edges += 1

        # Asignar cluster_id legible a cada componente conexo.
        root_to_cluster: Dict[int, str] = {}
        cluster_members: Dict[str, List[int]] = defaultdict(list)
        for idx in range(n_imgs):
            root = uf.find(idx)
            if root not in root_to_cluster:
                root_to_cluster[root] = f"cluster_{len(root_to_cluster):04d}"
            cid = root_to_cluster[root]
            records[idx]["cluster_id"] = cid
            cluster_members[cid].append(idx)

        n_clusters = len(cluster_members)
        logger.info(
            f"Deduplicación pHash completada: {n_imgs} imágenes -> {n_clusters} clústeres "
            f"(pares similares unidos={duplicate_edges})."
        )

        # Paso 3: Partición estratificada por clústeres (70% train / 15% val / 15% test).
        rng = random.Random(seed)
        clusters_by_label: Dict[str, List[str]] = defaultdict(list)
        for cid, indices in cluster_members.items():
            # Etiqueta predominante del clúster para estratificar de forma balanceada.
            lbl = Counter(records[i]["label"] for i in indices).most_common(1)[0][0]
            clusters_by_label[lbl].append(cid)

        cluster_to_split: Dict[str, str] = {}
        for lbl, cids in sorted(clusters_by_label.items()):
            shuffled = list(cids)
            rng.shuffle(shuffled)
            n_total = len(shuffled)
            n_val = max(1, round(0.15 * n_total))
            n_test = max(1, round(0.15 * n_total))
            n_train = max(1, n_total - n_val - n_test)
            for c in shuffled[:n_train]:
                cluster_to_split[c] = "train"
            for c in shuffled[n_train : n_train + n_val]:
                cluster_to_split[c] = "val"
            for c in shuffled[n_train + n_val :]:
                cluster_to_split[c] = "test"

        for rec in records:
            rec["split"] = cluster_to_split[str(rec["cluster_id"])]
            rec.pop("phash_bits", None)

        df = pd.DataFrame(records)

        # Paso 4: Generación sintética controlada de `vista no evaluable` DENTRO de cada split.
        # Esto asegura que ninguna imagen base de train genere una versión degradada en test.
        synth_dir = CACHE_DIR / "synthetic_non_evaluable"
        synth_rows: List[Dict[str, object]] = []
        modes = ["blur", "glare", "occlusion"]
        target_synth_per_split = {"train": 24, "val": 6, "test": 6}

        for split_name, n_synth in target_synth_per_split.items():
            pool = df[(df["split"] == split_name) & (df["label"] != "vista no evaluable")]
            if len(pool) == 0:
                continue
            chosen = pool.sample(
                n=min(n_synth, len(pool)),
                random_state=seed + len(split_name),
            )
            for idx_counter, (_, row) in enumerate(chosen.iterrows()):
                mode = modes[idx_counter % len(modes)]
                src_path = Path(str(row["image_path"]))
                out_name = f"synth_{split_name}_{mode}_{idx_counter:03d}_{src_path.stem}.jpg"
                out_path = synth_dir / split_name / out_name
                _create_synthetic_degradation(src_path, out_path, mode=mode, rng=rng)
                synth_rows.append(
                    {
                        "image_path": str(out_path.resolve()),
                        "raw_folder": f"Synthetic-{mode}",
                        "label": "vista no evaluable",
                        "phash_hex": str(row["phash_hex"]),
                        "width": 256,
                        "height": 256,
                        "is_synthetic": True,
                        "cluster_id": f"{row['cluster_id']}_synth",
                        "split": split_name,
                    }
                )

        if synth_rows:
            df = pd.concat([df, pd.DataFrame(synth_rows)], ignore_index=True)

        # Verificación estricta anti-fuga: ningún clúster base real puede aparecer en más de un split.
        real_df = df[~df["is_synthetic"]]
        cluster_splits = real_df.groupby("cluster_id")["split"].nunique()
        assert (cluster_splits == 1).all(), "Fuga detectada: un clúster pHash está en >1 split."

        splits_csv = SPLITS_DIR / "rgb_splits.csv"
        df.to_csv(splits_csv, index=False)

        summary = {
            "total_images": int(len(df)),
            "real_images": int(len(real_df)),
            "synthetic_non_evaluable": int(len(synth_rows)),
            "unique_clusters": int(n_clusters),
            "excluded_folders": dict(excluded_counts),
            "corrupted_files": corrupted_files,
            "class_distribution_by_split": {
                sp: {lbl: int(cnt) for lbl, cnt in sub["label"].value_counts().items()}
                for sp, sub in df.groupby("split")
            },
            "canonical_classes": RGB_CLASSES,
        }
        summary_path = SPLITS_DIR / "rgb_splits_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Manifiesto RGB guardado en {splits_csv} ({len(df)} muestras totales).")

        ctx["total_images"] = len(df)
        ctx["unique_clusters"] = n_clusters
        return df


if __name__ == "__main__":
    prepare_rgb_splits()
