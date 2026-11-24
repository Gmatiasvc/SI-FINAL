"""
Preparación de datos tabulares libre de fugas (No Data Leakage) para SolarGuard AI.

Paso a paso del pipeline:
1. Carga los registros crudos de generación eléctrica (`Plant_1_Generation_Data.csv`) y
   sensores meteorológicos (`Plant_1_Weather_Sensor_Data.csv`).
   Nota de auditoría: Se selecciona Planta 1 (~17,437 registros horarios) porque presenta
   alta integridad operativa (<0.03% de ceros diurnos por caída de inversor frente a 3.79%
   en Planta 2) y la columna `AC_POWER` está expresada consistentemente en kW en intervalos
   de 15 minutos, permitiendo derivar `energia_kwh_hora` de forma físicamente válida.
2. Parsea explícitamente los formatos de fecha heterogéneos (`%d-%m-%Y %H:%M` en generación
   y `%Y-%m-%d %H:%M:%S` en meteorología) antes de alinear por hora.
3. Agrega ambos conjuntos a granularidad horaria por `(PLANT_ID, SOURCE_KEY, fecha_hora)`
   calculando `energia_kwh_hora` como el promedio horario de `AC_POWER` (kW * 1 h = kWh/h),
   exigiendo al menos 2 lecturas válidas dentro de la hora para evitar horas truncadas.
4. Ejecuta partición temporal estricta por días consecutivos (prohibido el muestreo aleatorio):
   - Días 1 a 24  -> `train`
   - Días 25 a 29 -> `val`
   - Días 30 a 34 -> `test` (intacto hasta la evaluación final)
5. Calcula `capacidad_instalada` por `SOURCE_KEY` exclusivamente sobre el split `train`
   (percentil 99 de `energia_kwh_hora` en entrenamiento) y la mapea hacia `val` y `test`
   para impedir fuga de información estadística del futuro.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.core.config import CACHE_DIR, SPLITS_DIR, TABULAR_DATASET_DIR
from src.core.logging_config import get_logger
from src.core.telemetry import track_event

logger = get_logger(__name__)

# Columnas finales de entrada utilizadas por los modelos de Machine Learning.
TABULAR_FEATURE_COLUMNS: list[str] = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "hora",
    "dia_anio",
    "capacidad_instalada",
]
TABULAR_BASELINE_FEATURES: list[str] = ["IRRADIATION", "capacidad_instalada"]
TABULAR_TARGET_COLUMN: str = "energia_kwh_hora"


@dataclass
class TabularSplits:
    """Contenedor tipado con los tres conjuntos disjuntos y metadatos de auditoría."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    inverter_capacities: Dict[str, float]
    default_capacity: float
    summary: Dict[str, object]


def load_and_prepare_tabular_splits(
    dataset_dir: Path = TABULAR_DATASET_DIR,
    save_artifacts: bool = True,
) -> TabularSplits:
    """
    Ejecuta de forma reproducible toda la preparación tabular anti-fuga y persiste
    los metadatos del split temporal y capacidades por inversor.
    """
    with track_event(logger, "prepare_tabular_splits", dataset_dir=str(dataset_dir)) as ctx:
        gen_path = dataset_dir / "Plant_1_Generation_Data.csv"
        weather_path = dataset_dir / "Plant_1_Weather_Sensor_Data.csv"

        if not gen_path.exists() or not weather_path.exists():
            raise FileNotFoundError(
                f"No se encontraron los CSV esperados en {dataset_dir}. "
                "Se requieren Plant_1_Generation_Data.csv y Plant_1_Weather_Sensor_Data.csv."
            )

        # Paso 1: Lectura de archivos crudos.
        gen_df = pd.read_csv(gen_path)
        weather_df = pd.read_csv(weather_path)
        logger.info(
            f"Datos crudos cargados: generacion={len(gen_df)} filas, "
            f"meteorologia={len(weather_df)} filas."
        )

        # Paso 2: Parseo explícito de fechas según el formato real de cada archivo.
        gen_df["dt"] = pd.to_datetime(gen_df["DATE_TIME"], format="%d-%m-%Y %H:%M")
        weather_df["dt"] = pd.to_datetime(weather_df["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

        # Paso 3: Redondeo hacia abajo a la hora exacta para agregación horaria.
        gen_df["fecha_hora"] = gen_df["dt"].dt.floor("h")
        weather_df["fecha_hora"] = weather_df["dt"].dt.floor("h")

        # Paso 4: Agregación horaria de generación por inversor (`SOURCE_KEY`).
        # Como `AC_POWER` representa potencia media en kW cada 15 min, el promedio en la hora
        # multiplicado por 1 hora equivale directamente a `energia_kwh_hora`.
        gen_hourly = (
            gen_df.groupby(["PLANT_ID", "SOURCE_KEY", "fecha_hora"], as_index=False)
            .agg(
                energia_kwh_hora=("AC_POWER", "mean"),
                lecturas_hora=("AC_POWER", "count"),
            )
        )
        # Exigir al menos 2 mediciones en la hora para evitar sesgos de ventanas incompletas.
        before_filter = len(gen_hourly)
        gen_hourly = gen_hourly[gen_hourly["lecturas_hora"] >= 2].copy()
        logger.debug(
            f"Filtradas {before_filter - len(gen_hourly)} horas incompletas (<2 lecturas)."
        )

        # Paso 5: Agregación horaria de variables meteorológicas por planta.
        weather_hourly = (
            weather_df.groupby(["PLANT_ID", "fecha_hora"], as_index=False)
            .agg(
                IRRADIATION=("IRRADIATION", "mean"),
                AMBIENT_TEMPERATURE=("AMBIENT_TEMPERATURE", "mean"),
                MODULE_TEMPERATURE=("MODULE_TEMPERATURE", "mean"),
            )
        )

        # Paso 6: Fusión interna (inner join) por planta y hora exacta.
        merged_df = pd.merge(
            gen_hourly,
            weather_hourly,
            on=["PLANT_ID", "fecha_hora"],
            how="inner",
        )
        merged_df = merged_df.sort_values(["fecha_hora", "SOURCE_KEY"]).reset_index(drop=True)

        # Paso 7: Ingeniería de variables temporales conocidas antes/en el instante predicho.
        merged_df["hora"] = merged_df["fecha_hora"].dt.hour.astype(int)
        merged_df["dia_anio"] = merged_df["fecha_hora"].dt.dayofyear.astype(int)

        # Calcular día relativo consecutivo (1..N) respecto al primer día del dataset.
        first_day = merged_df["fecha_hora"].dt.normalize().min()
        merged_df["dia_relativo"] = (
            (merged_df["fecha_hora"].dt.normalize() - first_day).dt.days + 1
        ).astype(int)
        total_days = int(merged_df["dia_relativo"].max())

        # Paso 8: Partición estricta por bloques de días consecutivos (cero fuga temporal).
        # Días 1-24 -> train | Días 25-29 -> val | Días 30-34 -> test
        train_df = merged_df[merged_df["dia_relativo"] <= 24].copy()
        val_df = merged_df[
            (merged_df["dia_relativo"] >= 25) & (merged_df["dia_relativo"] <= 29)
        ].copy()
        test_df = merged_df[merged_df["dia_relativo"] >= 30].copy()

        # Paso 9: Derivar `capacidad_instalada` por `SOURCE_KEY` SOLO usando `train_df`.
        # Usar el percentil 99 en train evita outliers puntuales y jamás consulta val ni test.
        cap_series = train_df.groupby("SOURCE_KEY")[TABULAR_TARGET_COLUMN].quantile(0.99)
        inverter_capacities: Dict[str, float] = {
            str(k): round(float(v), 4) for k, v in cap_series.items()
        }
        default_capacity = round(float(cap_series.median()), 4)

        for split_df in (train_df, val_df, test_df):
            split_df["capacidad_instalada"] = (
                split_df["SOURCE_KEY"].map(inverter_capacities).fillna(default_capacity)
            )

        # Verificación explícita anti-fuga: las fechas máximas/mínimas no deben solaparse.
        assert train_df["fecha_hora"].max() < val_df["fecha_hora"].min(), (
            "Fuga temporal detectada entre train y val."
        )
        assert val_df["fecha_hora"].max() < test_df["fecha_hora"].min(), (
            "Fuga temporal detectada entre val y test."
        )

        summary: Dict[str, object] = {
            "source_files": ["Plant_1_Generation_Data.csv", "Plant_1_Weather_Sensor_Data.csv"],
            "total_merged_rows": int(len(merged_df)),
            "total_consecutive_days": total_days,
            "n_inverters": int(merged_df["SOURCE_KEY"].nunique()),
            "default_capacity_kwh": default_capacity,
            "splits": {
                "train": {
                    "days": "1-24",
                    "rows": int(len(train_df)),
                    "start": str(train_df["fecha_hora"].min()),
                    "end": str(train_df["fecha_hora"].max()),
                },
                "val": {
                    "days": "25-29",
                    "rows": int(len(val_df)),
                    "start": str(val_df["fecha_hora"].min()),
                    "end": str(val_df["fecha_hora"].max()),
                },
                "test": {
                    "days": f"30-{total_days}",
                    "rows": int(len(test_df)),
                    "start": str(test_df["fecha_hora"].min()),
                    "end": str(test_df["fecha_hora"].max()),
                },
            },
            "features": TABULAR_FEATURE_COLUMNS,
            "target": TABULAR_TARGET_COLUMN,
        }

        if save_artifacts:
            splits_json = SPLITS_DIR / "tabular_splits_info.json"
            splits_json.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            train_df.to_csv(CACHE_DIR / "tabular_train.csv", index=False)
            val_df.to_csv(CACHE_DIR / "tabular_val.csv", index=False)
            test_df.to_csv(CACHE_DIR / "tabular_test.csv", index=False)
            logger.info(f"Metadatos de split temporal guardados en {splits_json}")

        ctx["train_rows"] = len(train_df)
        ctx["val_rows"] = len(val_df)
        ctx["test_rows"] = len(test_df)
        return TabularSplits(
            train=train_df,
            val=val_df,
            test=test_df,
            inverter_capacities=inverter_capacities,
            default_capacity=default_capacity,
            summary=summary,
        )


if __name__ == "__main__":
    splits = load_and_prepare_tabular_splits()
    logger.info(f"Resumen tabular generado: {splits.summary['splits']}")
