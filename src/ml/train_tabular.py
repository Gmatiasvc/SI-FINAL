"""
Entrenamiento y evaluación comparativa del Experto 1 (Regresión Tabular de `energia_kwh_hora`).

Paso a paso:
1. Aplica `configure_cpu_limits()` para no bloquear la PC durante la búsqueda en Random Forest.
2. Carga los splits temporales estrictos (`train`: días 1-24, `val`: días 25-29, `test`: días 30-34).
3. Encapsula todo preprocesamiento (imputación de mediana + estandarización) dentro de
   `sklearn.pipeline.Pipeline` y `ColumnTransformer`, ajustándose únicamente con `train`.
4. Entrena y compara tres modelos:
   - Baseline: `LinearRegression` usando únicamente `IRRADIATION` y `capacidad_instalada`.
   - Candidato A: `RandomForestRegressor` con ajuste de hiperparámetros mediante
     `TimeSeriesSplit` respetando el orden cronológico dentro de `train`.
   - Candidato B: `GradientBoostingRegressor` con ajuste de hiperparámetros sobre `TimeSeriesSplit`.
5. Evalúa todos los modelos en `train`, `val` y `test` intacto con MAE (kWh), RMSE (kWh),
   nMAE (%) y nRMSE (%) normalizados por capacidad instalada promedio, más análisis por subgrupo.
6. Calibra la incertidumbre residual por hora del día sobre validación, guarda el modelo ganador
   en `artifacts/models/tabular_best_model.joblib`, las métricas en `artifacts/metrics/tabular_metrics.json`
   y genera gráficos comparativos en `report/figures/`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Usar backend Agg no interactivo para generar figuras sin requerir servidor gráfico X11.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.core.config import (  # noqa: E402
    FIGURES_DIR,
    MAX_CPU_THREADS,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    configure_cpu_limits,
)
from src.core.logging_config import get_logger, log_section  # noqa: E402
from src.core.telemetry import log_metric, track_event  # noqa: E402
from src.data.tabular_prep import (  # noqa: E402
    TABULAR_BASELINE_FEATURES,
    TABULAR_FEATURE_COLUMNS,
    TABULAR_TARGET_COLUMN,
    load_and_prepare_tabular_splits,
)

logger = get_logger(__name__)


def _build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """
    Construye un `ColumnTransformer` reproducible que imputa valores faltantes
    con la mediana y estandariza variables numéricas dentro del Pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop",
    )


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacities: np.ndarray,
) -> Dict[str, float]:
    """
    Calcula métricas obligatorias: MAE (kWh), RMSE (kWh), nMAE (%) y nRMSE (%)
    normalizados respecto a la capacidad instalada promedio del subconjunto.
    """
    # La potencia/energía fotovoltaica generada no puede ser negativa físicamente.
    y_pred_clipped = np.clip(y_pred, 0.0, None)
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(root_mean_squared_error(y_true, y_pred_clipped))
    mean_cap = float(np.mean(capacities)) if len(capacities) > 0 else 1.0
    nmae_pct = (mae / max(mean_cap, 1e-6)) * 100.0
    nrmse_pct = (rmse / max(mean_cap, 1e-6)) * 100.0
    return {
        "mae_kwh": round(mae, 3),
        "rmse_kwh": round(rmse, 3),
        "nmae_pct": round(nmae_pct, 3),
        "nrmse_pct": round(nrmse_pct, 3),
    }


def _evaluate_subgroups(
    df: pd.DataFrame,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Evalúa el rendimiento por subgrupos físicos relevantes en test:
    - Franja horaria: diurna (06:00-18:59) vs nocturna.
    - Nivel de irradiancia: baja (<0.2), media (0.2-0.6) y alta (>=0.6).
    """
    work = df.copy()
    work["pred"] = np.clip(y_pred, 0.0, None)

    subgroups: Dict[str, Dict[str, float]] = {}
    masks = {
        "franja_diurna_06_18": (work["hora"] >= 6) & (work["hora"] <= 18),
        "franja_nocturna": (work["hora"] < 6) | (work["hora"] > 18),
        "irradiancia_baja_lt_0.2": work["IRRADIATION"] < 0.2,
        "irradiancia_media_0.2_0.6": (work["IRRADIATION"] >= 0.2) & (work["IRRADIATION"] < 0.6),
        "irradiancia_alta_ge_0.6": work["IRRADIATION"] >= 0.6,
    }
    for name, mask in masks.items():
        sub = work[mask]
        if len(sub) == 0:
            continue
        metrics = _compute_regression_metrics(
            sub[TABULAR_TARGET_COLUMN].to_numpy(),
            sub["pred"].to_numpy(),
            sub["capacidad_instalada"].to_numpy(),
        )
        metrics["n_samples"] = int(len(sub))
        subgroups[name] = metrics
    return subgroups


def _generate_tabular_plots(
    results: Dict[str, Any],
    test_df: pd.DataFrame,
    best_preds: np.ndarray,
    best_name: str,
) -> None:
    """Genera los gráficos de comparación, dispersión real vs predicho y residuales horarios."""
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.3})

    # Gráfico 1: Comparativa MAE y RMSE en test entre Baseline, RandomForest y GradientBoosting.
    model_names = list(results["models"].keys())
    maes = [results["models"][m]["test"]["mae_kwh"] for m in model_names]
    rmses = [results["models"][m]["test"]["rmse_kwh"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - width / 2, maes, width, label="MAE (kWh)", color="#0E4D7A")
    bars2 = ax.bar(x + width / 2, rmses, width, label="RMSE (kWh)", color="#2E8B57")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=10)
    ax.set_ylabel("Error (kWh/h)")
    ax.set_title("Comparación de Modelos Tabulares en Test Temporal (Días 30-34)")
    ax.legend()
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tabular_model_comparison.png")
    plt.close(fig)

    # Gráfico 2: Valores reales vs predichos del modelo ganador en test.
    y_true = test_df[TABULAR_TARGET_COLUMN].to_numpy()
    y_pred = np.clip(best_preds, 0.0, None)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(
        y_true,
        y_pred,
        c=test_df["IRRADIATION"].to_numpy(),
        cmap="viridis",
        alpha=0.45,
        s=14,
    )
    max_val = float(max(np.max(y_true), np.max(y_pred)) * 1.03)
    ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Identidad ideal (y = x)")
    ax.set_xlabel("Energía real observada (kWh/h)")
    ax.set_ylabel("Energía esperada predicha (kWh/h)")
    ax.set_title(f"Predicción vs Realidad en Test — {best_name}")
    fig.colorbar(sc, ax=ax, label="Irradiancia normalizada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tabular_pred_vs_real.png")
    plt.close(fig)

    # Gráfico 3: Boxplot / media de residuales absolutos por hora del día en test.
    err_df = test_df[["hora"]].copy()
    err_df["abs_error"] = np.abs(y_true - y_pred)
    hourly_stats = err_df.groupby("hora")["abs_error"].agg(["mean", "std"]).fillna(0.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.errorbar(
        hourly_stats.index,
        hourly_stats["mean"],
        yerr=hourly_stats["std"],
        fmt="-o",
        color="#0E4D7A",
        ecolor="#A6C8E0",
        capsize=3,
        label="Error absoluto medio ± DE",
    )
    ax.set_xlabel("Hora del día (0–23)")
    ax.set_ylabel("Error absoluto (kWh/h)")
    ax.set_title(f"Distribución de Error Residual por Hora en Test — {best_name}")
    ax.set_xticks(range(0, 24))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "tabular_residuals_by_hour.png")
    plt.close(fig)


def train_and_evaluate_tabular() -> Dict[str, Any]:
    """
    Orquesta el entrenamiento secuencial del baseline y los dos modelos candidatos,
    guardando el mejor artefacto y todas las métricas auditables.
    """
    configure_cpu_limits(MAX_CPU_THREADS)
    log_section(logger, "ENTRENAMIENTO TABULAR ML (EXPERTO 1)")

    splits = load_and_prepare_tabular_splits(save_artifacts=True)
    train_df = splits.train.sort_values(["fecha_hora", "SOURCE_KEY"]).reset_index(drop=True)
    val_df = splits.val.sort_values(["fecha_hora", "SOURCE_KEY"]).reset_index(drop=True)
    test_df = splits.test.sort_values(["fecha_hora", "SOURCE_KEY"]).reset_index(drop=True)

    y_train = train_df[TABULAR_TARGET_COLUMN].to_numpy()
    y_val = val_df[TABULAR_TARGET_COLUMN].to_numpy()
    y_test = test_df[TABULAR_TARGET_COLUMN].to_numpy()

    cap_train = train_df["capacidad_instalada"].to_numpy()
    cap_val = val_df["capacidad_instalada"].to_numpy()
    cap_test = test_df["capacidad_instalada"].to_numpy()

    models_summary: Dict[str, Any] = {}
    trained_pipelines: Dict[str, Pipeline] = {}

    # ------------------------------------------------------------------
    # 1. BASELINE: LinearRegression (solo IRRADIATION y capacidad_instalada)
    # ------------------------------------------------------------------
    with track_event(logger, "train_tabular_baseline_linear"):
        baseline_pipe = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(TABULAR_BASELINE_FEATURES)),
                ("regressor", LinearRegression()),
            ]
        )
        baseline_pipe.fit(train_df[TABULAR_BASELINE_FEATURES], y_train)
        trained_pipelines["LinearRegression_Baseline"] = baseline_pipe

        models_summary["LinearRegression_Baseline"] = {
            "features": TABULAR_BASELINE_FEATURES,
            "best_params": {},
            "train": _compute_regression_metrics(
                y_train, baseline_pipe.predict(train_df[TABULAR_BASELINE_FEATURES]), cap_train
            ),
            "val": _compute_regression_metrics(
                y_val, baseline_pipe.predict(val_df[TABULAR_BASELINE_FEATURES]), cap_val
            ),
            "test": _compute_regression_metrics(
                y_test, baseline_pipe.predict(test_df[TABULAR_BASELINE_FEATURES]), cap_test
            ),
        }
        log_metric(
            logger,
            "tabular.LinearRegression_Baseline",
            models_summary["LinearRegression_Baseline"]["test"],
        )

    # Esquema de validación cruzada respetando el orden temporal dentro de `train`.
    tscv = TimeSeriesSplit(n_splits=4)

    # ------------------------------------------------------------------
    # 2. CANDIDATO A: RandomForestRegressor (con TimeSeriesSplit)
    # ------------------------------------------------------------------
    with track_event(logger, "train_tabular_random_forest", max_threads=MAX_CPU_THREADS) as ctx:
        rf_pipe = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(TABULAR_FEATURE_COLUMNS)),
                (
                    "regressor",
                    RandomForestRegressor(
                        random_state=RANDOM_SEED,
                        n_jobs=MAX_CPU_THREADS,
                    ),
                ),
            ]
        )
        rf_grid = {
            "regressor__n_estimators": [80, 140],
            "regressor__max_depth": [8, 12],
            "regressor__min_samples_leaf": [3, 6],
        }
        rf_search = GridSearchCV(
            rf_pipe,
            param_grid=rf_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,  # Paralelismo controlado dentro del RF (`n_jobs=MAX_CPU_THREADS`)
        )
        rf_search.fit(train_df[TABULAR_FEATURE_COLUMNS], y_train)
        best_rf = rf_search.best_estimator_
        trained_pipelines["RandomForestRegressor"] = best_rf

        models_summary["RandomForestRegressor"] = {
            "features": TABULAR_FEATURE_COLUMNS,
            "best_params": rf_search.best_params_,
            "cv_best_rmse": round(float(-rf_search.best_score_), 3),
            "train": _compute_regression_metrics(
                y_train, best_rf.predict(train_df[TABULAR_FEATURE_COLUMNS]), cap_train
            ),
            "val": _compute_regression_metrics(
                y_val, best_rf.predict(val_df[TABULAR_FEATURE_COLUMNS]), cap_val
            ),
            "test": _compute_regression_metrics(
                y_test, best_rf.predict(test_df[TABULAR_FEATURE_COLUMNS]), cap_test
            ),
        }
        ctx["best_params"] = rf_search.best_params_
        log_metric(
            logger, "tabular.RandomForestRegressor", models_summary["RandomForestRegressor"]["test"]
        )

    # ------------------------------------------------------------------
    # 3. CANDIDATO B: GradientBoostingRegressor (con TimeSeriesSplit)
    # ------------------------------------------------------------------
    with track_event(logger, "train_tabular_gradient_boosting") as ctx:
        gb_pipe = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(TABULAR_FEATURE_COLUMNS)),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        gb_grid = {
            "regressor__n_estimators": [120, 200],
            "regressor__learning_rate": [0.05, 0.1],
            "regressor__max_depth": [3, 4],
        }
        gb_search = GridSearchCV(
            gb_pipe,
            param_grid=gb_grid,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=MAX_CPU_THREADS,
        )
        gb_search.fit(train_df[TABULAR_FEATURE_COLUMNS], y_train)
        best_gb = gb_search.best_estimator_
        trained_pipelines["GradientBoostingRegressor"] = best_gb

        models_summary["GradientBoostingRegressor"] = {
            "features": TABULAR_FEATURE_COLUMNS,
            "best_params": gb_search.best_params_,
            "cv_best_rmse": round(float(-gb_search.best_score_), 3),
            "train": _compute_regression_metrics(
                y_train, best_gb.predict(train_df[TABULAR_FEATURE_COLUMNS]), cap_train
            ),
            "val": _compute_regression_metrics(
                y_val, best_gb.predict(val_df[TABULAR_FEATURE_COLUMNS]), cap_val
            ),
            "test": _compute_regression_metrics(
                y_test, best_gb.predict(test_df[TABULAR_FEATURE_COLUMNS]), cap_test
            ),
        }
        ctx["best_params"] = gb_search.best_params_
        log_metric(
            logger,
            "tabular.GradientBoostingRegressor",
            models_summary["GradientBoostingRegressor"]["test"],
        )

    # ------------------------------------------------------------------
    # Selección del mejor modelo basada en RMSE de VALIDACIÓN (no de test)
    # ------------------------------------------------------------------
    best_model_name = min(
        models_summary.keys(),
        key=lambda m: models_summary[m]["val"]["rmse_kwh"],
    )
    best_pipeline = trained_pipelines[best_model_name]
    best_features = models_summary[best_model_name]["features"]

    # Calibrar incertidumbre residual por hora del día usando las predicciones en `val`.
    val_preds = np.clip(best_pipeline.predict(val_df[best_features]), 0.0, None)
    val_residuals = val_df[["hora"]].copy()
    val_residuals["abs_err"] = np.abs(y_val - val_preds)
    residual_std_by_hour: Dict[int, float] = {
        int(h): round(float(val), 3)
        for h, val in val_residuals.groupby("hora")["abs_err"].std().fillna(5.0).items()
    }
    global_residual_std = round(float(np.std(y_val - val_preds)), 3)

    # Evaluar subgrupos físicos sobre el conjunto `test` intacto.
    test_preds = np.clip(best_pipeline.predict(test_df[best_features]), 0.0, None)
    subgroups = _evaluate_subgroups(test_df, test_preds)

    # Empaquetar modelo, esquema de características, capacidades e incertidumbre.
    artifact_bundle = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "features": best_features,
        "inverter_capacities": splits.inverter_capacities,
        "default_capacity": splits.default_capacity,
        "residual_std_by_hour": residual_std_by_hour,
        "global_residual_std": global_residual_std,
    }
    model_path = MODELS_DIR / "tabular_best_model.joblib"
    joblib.dump(artifact_bundle, model_path)
    logger.info(f"Modelo tabular ganador ({best_model_name}) guardado en {model_path}")

    full_report = {
        "best_model_selected_by_val_rmse": best_model_name,
        "split_summary": splits.summary,
        "models": models_summary,
        "test_subgroups": subgroups,
        "model_artifact": str(model_path),
    }

    metrics_path = METRICS_DIR / "tabular_metrics.json"
    metrics_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Métricas tabulares exportadas en {metrics_path}")

    _generate_tabular_plots(full_report, test_df, test_preds, best_model_name)
    logger.info("Gráficos de evaluación tabular generados en report/figures/.")
    return full_report


if __name__ == "__main__":
    train_and_evaluate_tabular()
