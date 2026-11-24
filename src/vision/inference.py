"""
Módulo unificado de inferencia visual para SolarGuard AI:
- Experto 2A: Detección y recorte espacial de paneles con YOLOv8n (`detectar_y_recortar_paneles`).
- Experto 2B: Clasificación RGB de 4 clases con regla de incertidumbre (`clasificar_panel`).
- Experto 3: Clasificación termográfica condicional IR de 3 clases (`clasificar_termografia`).

Paso a paso de seguridad y calidad:
1. Valida la existencia, extensión y legibilidad de la imagen antes de inferir.
2. Si YOLO detecta cajas de `single_panel` o `double_panel`, recorta los paneles detectados;
   si la foto ya muestra un primer plano de un único panel y YOLO no detecta cajas, usa
   como respaldo controlado la imagen completa para no perder capacidad de diagnóstico.
3. Si el clasificador RGB arroja `vista no evaluable` o su confianza máxima es `< 0.60`,
   marca `es_evaluable = False` y fuerza la etiqueta operativa `vista no evaluable` para que
   el grafo LangGraph active `uncertainty_flag = True`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.core.config import (
    CACHE_DIR,
    MAX_CPU_THREADS,
    MODELS_DIR,
    RGB_CLASSES,
    THERMAL_CLASSES,
    VISION_CONFIDENCE_THRESHOLD,
    YOLO_CONFIDENCE_THRESHOLD,
    configure_cpu_limits,
)
from src.core.logging_config import get_logger
from src.core.telemetry import track_event

logger = get_logger(__name__)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class VisionInputError(ValueError):
    """Error lanzado cuando la ruta o formato de imagen visual/térmica es inválido."""


def _validate_image_file(image_path: str | Path) -> Path:
    """Verifica existencia, extensión permitida y decodificación PIL de la imagen."""
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        raise VisionInputError(f"No se encontró el archivo de imagen: {p}")
    if p.suffix.lower() not in VALID_EXTS:
        raise VisionInputError(
            f"Formato de imagen no soportado '{p.suffix}'. Use {sorted(VALID_EXTS)}."
        )
    try:
        with Image.open(p) as im:
            im.verify()
    except Exception as exc:
        raise VisionInputError(f"Archivo de imagen corrupto o ilegible ({p}): {exc}") from exc
    return p


@lru_cache(maxsize=2)
def _load_rgb_model(ckpt_path_str: str = str(MODELS_DIR / "rgb_classifier.pt")) -> Dict[str, Any]:
    """Carga perezosa del checkpoint RGB MobileNetV3-Small."""
    configure_cpu_limits(MAX_CPU_THREADS)
    ckpt_path = Path(ckpt_path_str)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Modelo RGB no encontrado en {ckpt_path}. Ejecute `python -m src.vision.train_rgb`."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes: List[str] = ckpt.get("classes", RGB_CLASSES)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tfms = transforms.Compose(
        [
            transforms.Resize((int(ckpt.get("input_size", 224)), int(ckpt.get("input_size", 224)))),
            transforms.ToTensor(),
            transforms.Normalize(ckpt["mean"], ckpt["std"]),
        ]
    )
    return {"model": model, "classes": classes, "tfms": tfms}


@lru_cache(maxsize=2)
def _load_thermal_model(
    ckpt_path_str: str = str(MODELS_DIR / "thermal_classifier.pt"),
) -> Dict[str, Any]:
    """Carga perezosa del checkpoint térmico IR MobileNetV3-Small."""
    configure_cpu_limits(MAX_CPU_THREADS)
    ckpt_path = Path(ckpt_path_str)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Modelo IR no encontrado en {ckpt_path}. Ejecute `python -m src.vision.train_thermal`."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes: List[str] = ckpt.get("classes", THERMAL_CLASSES)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    sz = int(ckpt.get("input_size", 128))
    tfms = transforms.Compose(
        [
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize(ckpt["mean"], ckpt["std"]),
        ]
    )
    return {"model": model, "classes": classes, "tfms": tfms}


@lru_cache(maxsize=2)
def _load_yolo_model(ckpt_path_str: str = str(MODELS_DIR / "yolo_panel_detector.pt")) -> Any:
    """Carga perezosa del detector YOLOv8n de paneles."""
    from ultralytics import YOLO

    configure_cpu_limits(MAX_CPU_THREADS)
    ckpt_path = Path(ckpt_path_str)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Modelo YOLO no encontrado en {ckpt_path}. Ejecute `python -m src.vision.train_yolo`."
        )
    return YOLO(str(ckpt_path))


def detectar_y_recortar_paneles(
    ruta_imagen: str | Path,
    conf: float = YOLO_CONFIDENCE_THRESHOLD,
    max_crops: int = 3,
) -> Dict[str, Any]:
    """
    Experto 2A: Ejecuta YOLOv8n sobre `ruta_imagen` y guarda recortes de los paneles
    detectados en `artifacts/cache/crops/`. Si no se detectan cajas (ej. foto en primer plano),
    retorna la ruta original como recorte único de respaldo.
    """
    img_path = _validate_image_file(ruta_imagen)
    with track_event(logger, "yolo_detect_panels", image=img_path.name) as ctx:
        yolo_path = MODELS_DIR / "yolo_panel_detector.pt"
        if not yolo_path.exists():
            logger.warning("Checkpoint YOLO no encontrado; usando imagen completa como fallback.")
            return {
                "crops": [str(img_path.resolve())],
                "detections": [],
                "used_fallback_full_image": True,
            }

        detector = _load_yolo_model(str(yolo_path))
        result = detector.predict(
            source=str(img_path),
            imgsz=320,
            conf=conf,
            device="cpu",
            verbose=False,
        )[0]

        crops_dir = CACHE_DIR / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        detections: List[Dict[str, Any]] = []
        crop_paths: List[str] = []
        names = result.names

        with Image.open(img_path) as im:
            rgb_im = im.convert("RGB")
            w_img, h_img = rgb_im.size

            if result.boxes is not None and len(result.boxes) > 0:
                # Ordenar cajas por confianza descendente.
                confs = result.boxes.conf.cpu().numpy()
                order = np.argsort(confs)[::-1]
                for rank, box_idx in enumerate(order[:max_crops]):
                    box = result.boxes[int(box_idx)]
                    cls_id = int(box.cls.item())
                    cls_name = str(names.get(cls_id, cls_id))
                    box_conf = round(float(box.conf.item()), 4)
                    x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)

                    # Filtrar cajas demasiado pequeñas (< 24px) o etiquetadas como 'dummy'.
                    if (x2 - x1) < 24 or (y2 - y1) < 24 or cls_name == "dummy":
                        continue

                    crop_file = crops_dir / f"{img_path.stem}_crop_{rank}.jpg"
                    rgb_im.crop((x1, y1, x2, y2)).save(crop_file, format="JPEG", quality=92)
                    crop_paths.append(str(crop_file.resolve()))
                    detections.append(
                        {
                            "class": cls_name,
                            "confidence": box_conf,
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "crop_path": str(crop_file.resolve()),
                        }
                    )

        used_fallback = False
        if not crop_paths:
            used_fallback = True
            crop_paths = [str(img_path.resolve())]

        ctx["n_detections"] = len(detections)
        ctx["used_fallback"] = used_fallback
        return {
            "crops": crop_paths,
            "detections": detections,
            "used_fallback_full_image": used_fallback,
        }


def clasificar_panel(
    ruta_imagen: str | Path,
    run_yolo: bool = True,
    confidence_threshold: float = VISION_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Experto 2A + 2B: Localiza paneles (opcionalmente con YOLO) y clasifica el estado
    superficial RGB en las 4 clases canónicas.

    Regla de incertidumbre obligatoria:
    Si la clase predicha es `vista no evaluable` o la confianza es `< confidence_threshold` (0.60),
    `es_evaluable` se fija en `False` y se exige recaptura fotográfica.
    """
    img_path = _validate_image_file(ruta_imagen)
    with track_event(logger, "classify_rgb_panel", image=img_path.name, run_yolo=run_yolo) as ctx:
        bundle = _load_rgb_model()
        model: nn.Module = bundle["model"]
        classes: List[str] = bundle["classes"]
        tfms = bundle["tfms"]

        yolo_info = (
            detectar_y_recortar_paneles(img_path)
            if run_yolo
            else {
                "crops": [str(img_path.resolve())],
                "detections": [],
                "used_fallback_full_image": True,
            }
        )
        crop_paths: List[str] = yolo_info["crops"]

        # Clasificar cada recorte (y también evaluar el encuadre principal).
        crop_results: List[Dict[str, Any]] = []
        with torch.no_grad():
            for c_path in crop_paths:
                with Image.open(c_path) as im:
                    tensor = tfms(im.convert("RGB")).unsqueeze(0)
                probs = torch.softmax(model(tensor), dim=1).squeeze(0).numpy()
                top_idx = int(np.argmax(probs))
                raw_label = classes[top_idx]
                conf = round(float(probs[top_idx]), 4)
                prob_dict = {cls: round(float(probs[i]), 4) for i, cls in enumerate(classes)}
                crop_results.append(
                    {
                        "crop_path": c_path,
                        "raw_label": raw_label,
                        "confidence": conf,
                        "all_probs": prob_dict,
                    }
                )

        # Seleccionar el hallazgo prioritario: si algún recorte muestra grieta o suciedad evaluable,
        # priorizar severidad (`grieta visible` > `suciedad visible` > `panel limpio` > `vista no evaluable`).
        severity_rank = {
            "grieta visible": 3,
            "suciedad visible": 2,
            "panel limpio": 1,
            "vista no evaluable": 0,
        }
        primary = max(
            crop_results,
            key=lambda r: (
                1 if r["confidence"] >= confidence_threshold else 0,
                severity_rank.get(r["raw_label"], 0),
                r["confidence"],
            ),
        )

        final_label = primary["raw_label"]
        final_conf = primary["confidence"]
        es_evaluable = (final_label != "vista no evaluable") and (final_conf >= confidence_threshold)
        if not es_evaluable and final_label != "vista no evaluable":
            # Si la confianza es menor al umbral 0.60, escalar a estado no evaluable por seguridad.
            final_label = "vista no evaluable"

        ctx["label"] = final_label
        ctx["confidence"] = final_conf
        ctx["es_evaluable"] = es_evaluable

        return {
            "image_path": str(img_path.resolve()),
            "label": final_label,
            "raw_label": primary["raw_label"],
            "confidence": final_conf,
            "all_probs": primary["all_probs"],
            "es_evaluable": es_evaluable,
            "confidence_threshold": confidence_threshold,
            "detected_panels_crops": crop_paths,
            "yolo_detections": yolo_info["detections"],
            "crop_predictions": crop_results,
        }


def clasificar_termografia(ruta_imagen_ir: str | Path) -> Dict[str, Any]:
    """
    Experto 3 (Condicional): Clasifica una captura termográfica IR en
    `Normal térmico`, `Hotspot / Celda crítica` o `Fallo de diodo / Bypass`.
    """
    img_path = _validate_image_file(ruta_imagen_ir)
    with track_event(logger, "classify_thermal_ir", image=img_path.name) as ctx:
        bundle = _load_thermal_model()
        model: nn.Module = bundle["model"]
        classes: List[str] = bundle["classes"]
        tfms = bundle["tfms"]

        with torch.no_grad():
            with Image.open(img_path) as im:
                tensor = tfms(im.convert("RGB")).unsqueeze(0)
            probs = torch.softmax(model(tensor), dim=1).squeeze(0).numpy()
            top_idx = int(np.argmax(probs))
            label = classes[top_idx]
            conf = round(float(probs[top_idx]), 4)
            prob_dict = {cls: round(float(probs[i]), 4) for i, cls in enumerate(classes)}

        ctx["thermal_label"] = label
        ctx["confidence"] = conf
        return {
            "image_path": str(img_path.resolve()),
            "label": label,
            "confidence": conf,
            "all_probs": prob_dict,
        }
