"""
Entrenamiento y evaluación del Experto 2B: Clasificador Visual RGB (4 clases obligatorias).

Clases:
0: `panel limpio`
1: `suciedad visible`
2: `grieta visible`
3: `vista no evaluable`

Paso a paso:
1. Limita hilos de CPU (`configure_cpu_limits`) y fuerza `num_workers=0` para evitar
   saturar memoria o bloquear la PC en CPU.
2. Carga el manifiesto deduplicado por pHash (`artifacts/splits/rgb_splits.csv`).
3. Compara dos modelos bajo las mismas particiones disjuntas:
   - Baseline sin ajuste (`MobileNetV3-Small_Frozen`): backbone preentrenado congelado
     y solo capa clasificadora entrenada durante 5 épocas.
   - Modelo principal con Transfer Learning (`MobileNetV3-Small_FineTuned`): ajuste fino
     de los últimos bloques convolucionales + cabeza clasificadora con pérdida ponderada
     inversamente a la frecuencia de clase para proteger el Recall de `grieta visible`.
4. Selecciona el mejor punto de control mediante `Macro-F1` en validación y evalúa
   sobre el conjunto `test` intacto reportando `Macro-F1`, `Recall` por clase, matriz de
   confusión y galería visual de errores / casos difíciles.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
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
    RGB_CLASSES,
    SPLITS_DIR,
    configure_cpu_limits,
)
from src.core.logging_config import get_logger, log_section  # noqa: E402
from src.core.telemetry import log_metric, track_event  # noqa: E402
from src.data.vision_prep import prepare_rgb_splits  # noqa: E402

logger = get_logger(__name__)

# Estadísticas estándar ImageNet para modelos preentrenados en torchvision.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PanelImageDataset(Dataset):
    """Dataset PyTorch ligero que lee imágenes RGB bajo demanda desde el manifiesto."""

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


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_mobilenet_rgb(num_classes: int, freeze_backbone: bool) -> nn.Module:
    """
    Construye `mobilenet_v3_small` preentrenado en ImageNet.
    Si `freeze_backbone=True`, congela todos los pesos convolucionales (baseline).
    Si `freeze_backbone=False`, descongela los últimos 4 bloques de `features`
    para adaptar texturas solares reales manteniendo regularización.
    """
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    except Exception as exc:
        logger.warning(
            f"No se pudieron cargar pesos ImageNet preentrenados ({exc}); "
            "inicializando sin pesos remotos."
        )
        model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    # Siempre entrenar la cabeza clasificadora.
    for param in model.classifier.parameters():
        param.requires_grad = True

    if not freeze_backbone:
        # Descongelar los últimos 4 bloques de extracción de características.
        for block in list(model.features.children())[-4:]:
            for param in block.parameters():
                param.requires_grad = True

    return model


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ejecuta inferencia sin gradientes y retorna (y_true, y_pred, probs)."""
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_true.extend(y_batch.numpy().tolist())
            all_pred.extend(preds.tolist())
            all_probs.append(probs)

    probs_cat = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, len(RGB_CLASSES)))
    return np.array(all_true), np.array(all_pred), probs_cat


def _train_model_variant(
    variant_name: str,
    freeze_backbone: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, List[float]], float]:
    """Entrena una variante del clasificador RGB seleccionando los pesos con mejor Macro-F1 en val."""
    model = _build_mobilenet_rgb(len(RGB_CLASSES), freeze_backbone=freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    history: Dict[str, List[float]] = {"train_loss": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience = 4
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(y_batch)
            n_samples += len(y_batch)

        scheduler.step()
        avg_loss = running_loss / max(1, n_samples)

        y_val_true, y_val_pred, _ = _evaluate_loader(model, val_loader, device)
        val_f1 = float(f1_score(y_val_true, y_val_pred, average="macro", zero_division=0))
        history["train_loss"].append(round(avg_loss, 4))
        history["val_macro_f1"].append(round(val_f1, 4))

        logger.info(
            f"[{variant_name}] Época {epoch:02d}/{epochs:02d} | "
            f"train_loss={avg_loss:.4f} | val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[{variant_name}] Early stopping activado en época {epoch}.")
                break

    model.load_state_dict(best_state)
    return model, history, best_macro_f1


def _save_rgb_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    test_df: pd.DataFrame,
    history_ft: Dict[str, List[float]],
) -> None:
    """Genera matriz de confusión, curvas de entrenamiento y galería de errores en `report/figures/`."""
    # 1. Matriz de confusión en test.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(RGB_CLASSES))))
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=RGB_CLASSES,
        yticklabels=RGB_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicción del modelo")
    ax.set_ylabel("Etiqueta real")
    ax.set_title("Matriz de Confusión RGB en Test Disjunto (MobileNetV3-Small)")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rgb_confusion_matrix.png", dpi=140)
    plt.close(fig)

    # 2. Curvas de entrenamiento (pérdida y Macro-F1 en validación).
    epochs_range = list(range(1, len(history_ft["train_loss"]) + 1))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.plot(epochs_range, history_ft["train_loss"], "-o", color="#0E4D7A", label="Train Loss")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Cross-Entropy Loss", color="#0E4D7A")
    ax2 = ax1.twinx()
    ax2.plot(
        epochs_range, history_ft["val_macro_f1"], "-s", color="#2E8B57", label="Val Macro-F1"
    )
    ax2.set_ylabel("Validation Macro-F1", color="#2E8B57")
    ax1.set_title("Curvas de Entrenamiento — Clasificador Visual RGB (Fine-Tuning)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rgb_training_curves.png", dpi=140)
    plt.close(fig)

    # 3. Galería de errores / casos difíciles en test (hasta 6 ejemplos).
    mis_indices = np.where(y_true != y_pred)[0]
    if len(mis_indices) == 0:
        # Si no hay errores, mostrar los casos con menor confianza como casos difíciles.
        confs = np.max(probs, axis=1)
        selected_indices = np.argsort(confs)[:6]
        title = "Casos con Menor Confianza en Test RGB"
    else:
        selected_indices = mis_indices[:6]
        title = "Galería de Errores en Test Disjunto RGB"

    n_show = len(selected_indices)
    cols = min(3, max(1, n_show))
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax_i, ax in enumerate(axes_arr):
        if ax_i >= n_show:
            ax.axis("off")
            continue
        idx = int(selected_indices[ax_i])
        img_path = test_df.iloc[idx]["image_path"]
        with Image.open(img_path) as im:
            ax.imshow(im.convert("RGB").resize((224, 224)))
        true_lbl = RGB_CLASSES[int(y_true[idx])]
        pred_lbl = RGB_CLASSES[int(y_pred[idx])]
        conf = float(np.max(probs[idx]))
        ax.set_title(f"Real: {true_lbl}\nPred: {pred_lbl} ({conf:.2f})", fontsize=8.5)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rgb_error_gallery.png", dpi=140)
    plt.close(fig)


def train_and_evaluate_rgb() -> Dict[str, Any]:
    """Ejecuta el entrenamiento secuencial del clasificador RGB de 4 clases."""
    configure_cpu_limits(MAX_CPU_THREADS)
    _set_seed(RANDOM_SEED)
    log_section(logger, "ENTRENAMIENTO VISIÓN RGB — 4 CLASES (EXPERTO 2B)")

    splits_csv = SPLITS_DIR / "rgb_splits.csv"
    df = pd.read_csv(splits_csv) if splits_csv.exists() else prepare_rgb_splits()

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    class_to_idx = {name: i for i, name in enumerate(RGB_CLASSES)}

    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = PanelImageDataset(train_df, class_to_idx, train_tfms)
    val_ds = PanelImageDataset(val_df, class_to_idx, eval_tfms)
    test_ds = PanelImageDataset(test_df, class_to_idx, eval_tfms)

    # num_workers=0 es obligatorio en CPU para no multiplicar memoria ni bloquear procesos.
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    # Calcular pesos balanceados inversos suavizados (raíz cuadrada) para no perder Recall en grietas.
    counts = np.array([sum(train_df["label"] == c) for c in RGB_CLASSES], dtype=np.float32)
    raw_weights = np.sum(counts) / (len(RGB_CLASSES) * np.maximum(counts, 1.0))
    smooth_weights = torch.tensor(np.sqrt(raw_weights), dtype=torch.float32)

    device = torch.device("cpu")

    # 1. Baseline sin ajuste fino (backbone congelado).
    with track_event(logger, "train_rgb_baseline_frozen", train_n=len(train_ds)) as ctx:
        baseline_model, hist_base, val_f1_base = _train_model_variant(
            "MobileNetV3_Frozen_Baseline",
            freeze_backbone=True,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=smooth_weights,
            epochs=5,
            lr=1e-3,
            device=device,
        )
        y_true_b, y_pred_b, _ = _evaluate_loader(baseline_model, test_loader, device)
        rep_base = classification_report(
            y_true_b, y_pred_b, target_names=RGB_CLASSES, output_dict=True, zero_division=0
        )
        ctx["test_macro_f1"] = round(float(rep_base["macro avg"]["f1-score"]), 4)

    # 2. Modelo principal con Transfer Learning (fine-tuning de los últimos bloques).
    with track_event(logger, "train_rgb_finetuned", train_n=len(train_ds)) as ctx:
        ft_model, hist_ft, val_f1_ft = _train_model_variant(
            "MobileNetV3_Small_FineTuned",
            freeze_backbone=False,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=smooth_weights,
            epochs=12,
            lr=4e-4,
            device=device,
        )
        y_true_ft, y_pred_ft, probs_ft = _evaluate_loader(ft_model, test_loader, device)
        rep_ft = classification_report(
            y_true_ft, y_pred_ft, target_names=RGB_CLASSES, output_dict=True, zero_division=0
        )
        ctx["test_macro_f1"] = round(float(rep_ft["macro avg"]["f1-score"]), 4)

    # Seleccionar ganador según Macro-F1 de validación.
    if val_f1_ft >= val_f1_base:
        winner_name = "MobileNetV3_Small_FineTuned"
        winner_model = ft_model
        y_true_win, y_pred_win, probs_win, rep_win = y_true_ft, y_pred_ft, probs_ft, rep_ft
    else:
        winner_name = "MobileNetV3_Small_Frozen_Baseline"
        winner_model = baseline_model
        _, _, probs_base = _evaluate_loader(baseline_model, test_loader, device)
        y_true_win, y_pred_win, probs_win, rep_win = y_true_b, y_pred_b, probs_base, rep_base

    checkpoint_path = MODELS_DIR / "rgb_classifier.pt"
    torch.save(
        {
            "arch": "mobilenet_v3_small",
            "model_name": winner_name,
            "classes": RGB_CLASSES,
            "input_size": 224,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "state_dict": winner_model.state_dict(),
        },
        checkpoint_path,
    )
    logger.info(f"Pesos RGB ganadores ({winner_name}) guardados en {checkpoint_path}")

    # Analizar errores más frecuentes para documentarlos en el informe.
    misclassified_samples: List[Dict[str, Any]] = []
    for i in np.where(y_true_win != y_pred_win)[0][:10]:
        misclassified_samples.append(
            {
                "image": Path(str(test_df.iloc[int(i)]["image_path"])).name,
                "true_label": RGB_CLASSES[int(y_true_win[i])],
                "pred_label": RGB_CLASSES[int(y_pred_win[i])],
                "confidence": round(float(np.max(probs_win[i])), 4),
            }
        )

    metrics_payload = {
        "winner_model": winner_name,
        "split_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "baseline_frozen": {
            "val_macro_f1": round(val_f1_base, 4),
            "test_macro_f1": round(float(rep_base["macro avg"]["f1-score"]), 4),
            "test_accuracy": round(float(rep_base["accuracy"]), 4),
            "per_class": rep_base,
        },
        "finetuned": {
            "val_macro_f1": round(val_f1_ft, 4),
            "test_macro_f1": round(float(rep_ft["macro avg"]["f1-score"]), 4),
            "test_accuracy": round(float(rep_ft["accuracy"]), 4),
            "per_class": rep_ft,
            "history": hist_ft,
        },
        "damage_recall_summary": {
            "suciedad_visible_recall": round(float(rep_win["suciedad visible"]["recall"]), 4),
            "grieta_visible_recall": round(float(rep_win["grieta visible"]["recall"]), 4),
            "panel_limpio_recall": round(float(rep_win["panel limpio"]["recall"]), 4),
            "vista_no_evaluable_recall": round(float(rep_win["vista no evaluable"]["recall"]), 4),
        },
        "confusion_matrix": confusion_matrix(
            y_true_win, y_pred_win, labels=list(range(len(RGB_CLASSES)))
        ).tolist(),
        "misclassified_examples": misclassified_samples,
        "model_artifact": str(checkpoint_path),
    }

    metrics_path = METRICS_DIR / "rgb_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_metric(
        logger,
        "vision.rgb_classifier",
        {
            "winner": winner_name,
            "test_macro_f1": metrics_payload["finetuned"]["test_macro_f1"],
            **metrics_payload["damage_recall_summary"],
        },
    )

    _save_rgb_plots(y_true_win, y_pred_win, probs_win, test_df, hist_ft)
    logger.info("Gráficos de evaluación RGB guardados en report/figures/.")
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate_rgb()
