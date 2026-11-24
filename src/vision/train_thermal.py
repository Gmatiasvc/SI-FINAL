"""
Entrenamiento y evaluación del Experto 3: Clasificador Termográfico IR (3 clases).

Clases:
0: `Normal térmico`
1: `Hotspot / Celda crítica`
2: `Fallo de diodo / Bypass`

Paso a paso:
1. Limita hilos de CPU (`configure_cpu_limits`) y usa `num_workers=0` para ejecución estable.
2. Carga la submuestra balanceada de 2,000 imágenes IR (`artifacts/splits/thermal_splits.csv`).
3. Escala los parches infrarrojos a 128x128 RGB y entrena `MobileNetV3-Small` con Transfer
   Learning (descongelando los últimos 4 bloques y cabeza clasificadora), optimizador AdamW
   y early stopping sobre `Macro-F1` de validación.
4. Evalúa en `test` intacto reportando Macro-F1, Recall por clase, matriz de confusión
   (`report/figures/thermal_confusion_matrix.png`) y curvas (`report/figures/thermal_training_curves.png`).
5. Guarda pesos en `artifacts/models/thermal_classifier.pt` y métricas en
   `artifacts/metrics/thermal_metrics.json`.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src.core.config import (  # noqa: E402
    FIGURES_DIR,
    MAX_CPU_THREADS,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    SPLITS_DIR,
    THERMAL_CLASSES,
    configure_cpu_limits,
)
from src.core.logging_config import get_logger, log_section  # noqa: E402
from src.core.telemetry import log_metric, track_event  # noqa: E402
from src.data.thermal_prep import prepare_thermal_splits  # noqa: E402

logger = get_logger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ThermalDataset(Dataset):
    """Dataset ligero para parches termográficos IR."""

    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: Dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.paths = df["image_path"].tolist()
        self.labels = [class_to_idx[lbl] for lbl in df["label"].tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        with Image.open(self.paths[idx]) as im:
            img = im.convert("RGB")
        return self.transform(img), self.labels[idx]


def _build_thermal_model(num_classes: int = 3) -> nn.Module:
    """Construye `mobilenet_v3_small` adaptado a clasificación de anomalías térmicas."""
    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    # Descongelar los últimos 5 bloques para adaptar patrones de gradientes térmicos.
    for block in list(model.features.children())[-5:]:
        for param in block.parameters():
            param.requires_grad = True
    return model


def _eval_thermal(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    trues, preds, probs_list = [], [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            logits = model(x_b.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            trues.extend(y_b.numpy().tolist())
            preds.extend(np.argmax(probs, axis=1).tolist())
            probs_list.append(probs)
    probs_cat = np.concatenate(probs_list, axis=0) if probs_list else np.empty((0, 3))
    return np.array(trues), np.array(preds), probs_cat


def train_and_evaluate_thermal() -> Dict[str, Any]:
    """Entrena y evalúa de forma secuencial el clasificador térmico IR de 3 clases."""
    configure_cpu_limits(MAX_CPU_THREADS)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    log_section(logger, "ENTRENAMIENTO TERMOGRAFÍA IR — 3 CLASES (EXPERTO 3)")

    splits_csv = SPLITS_DIR / "thermal_splits.csv"
    df = pd.read_csv(splits_csv) if splits_csv.exists() else prepare_thermal_splits()

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    class_to_idx = {name: i for i, name in enumerate(THERMAL_CLASSES)}
    input_size = 128  # Parches nativos son 24x40; 128x128 preserva detalles y acelera CPU.

    train_tfms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_loader = DataLoader(
        ThermalDataset(train_df, class_to_idx, train_tfms),
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ThermalDataset(val_df, class_to_idx, eval_tfms),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        ThermalDataset(test_df, class_to_idx, eval_tfms),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cpu")
    model = _build_thermal_model(len(THERMAL_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4
    )
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {"train_loss": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience = 4
    no_improve = 0

    with track_event(logger, "train_thermal_mobilenet", train_n=len(train_df)) as ctx:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0
            for x_b, y_b in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(x_b.to(device)), y_b.to(device))
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(y_b)
                n_seen += len(y_b)
            scheduler.step()

            avg_loss = running_loss / max(1, n_seen)
            y_vt, y_vp, _ = _eval_thermal(model, val_loader, device)
            val_f1 = float(f1_score(y_vt, y_vp, average="macro", zero_division=0))
            history["train_loss"].append(round(avg_loss, 4))
            history["val_macro_f1"].append(round(val_f1, 4))
            logger.info(
                f"[MobileNetV3_Thermal] Época {epoch:02d}/{epochs:02d} | "
                f"train_loss={avg_loss:.4f} | val_macro_f1={val_f1:.4f}"
            )
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"[MobileNetV3_Thermal] Early stopping en época {epoch}.")
                    break

        model.load_state_dict(best_state)
        y_true, y_pred, _ = _eval_thermal(model, test_loader, device)
        report = classification_report(
            y_true, y_pred, target_names=THERMAL_CLASSES, output_dict=True, zero_division=0
        )
        ctx["test_macro_f1"] = round(float(report["macro avg"]["f1-score"]), 4)

    checkpoint_path = MODELS_DIR / "thermal_classifier.pt"
    torch.save(
        {
            "arch": "mobilenet_v3_small_ir",
            "classes": THERMAL_CLASSES,
            "input_size": input_size,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    logger.info(f"Modelo térmico IR guardado en {checkpoint_path}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(THERMAL_CLASSES))))
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=THERMAL_CLASSES,
        yticklabels=THERMAL_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicción IR")
    ax.set_ylabel("Etiqueta IR real")
    ax.set_title("Matriz de Confusión Termografía IR (Test = 300 parches)")
    plt.xticks(rotation=15, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "thermal_confusion_matrix.png", dpi=140)
    plt.close(fig)

    ep = list(range(1, len(history["train_loss"]) + 1))
    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
    ax1.plot(ep, history["train_loss"], "-o", color="#B2471B", label="Train Loss")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss", color="#B2471B")
    ax2 = ax1.twinx()
    ax2.plot(ep, history["val_macro_f1"], "-s", color="#0E4D7A", label="Val Macro-F1")
    ax2.set_ylabel("Val Macro-F1", color="#0E4D7A")
    ax1.set_title("Entrenamiento Termografía IR — MobileNetV3-Small")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "thermal_training_curves.png", dpi=140)
    plt.close(fig)

    metrics_payload = {
        "model_name": "MobileNetV3_Small_Thermal_IR",
        "split_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "val_macro_f1": round(best_f1, 4),
        "test_macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "test_accuracy": round(float(report["accuracy"]), 4),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "model_artifact": str(checkpoint_path),
    }

    metrics_path = METRICS_DIR / "thermal_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_metric(
        logger,
        "vision.thermal_classifier",
        {
            "test_macro_f1": metrics_payload["test_macro_f1"],
            "test_accuracy": metrics_payload["test_accuracy"],
        },
    )
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate_thermal()
