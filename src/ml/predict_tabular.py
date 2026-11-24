"""
Inferencia y cálculo de discrepancia energética del Experto 1 (Regresión Tabular).

Paso a paso:
1. Carga perezosa (`lazy loading`) del artefacto `artifacts/models/tabular_best_model.joblib`.
2. Valida estrictamente los rangos físicos y tipos de las variables meteorológicas y operativas
   (irradiancia >= 0, temperaturas plausibles, hora 0-23, día del año 1-366, capacidad > 0).
3. Ejecuta la predicción de `energia_kwh_hora` esperada y estima la incertidumbre (kWh)
   combinando la varianza entre estimadores (si el modelo ganador es RandomForest) y los
   residuales históricos por franja horaria calibrados en validación.
4. Calcula la discrepancia porcentual respecto a `energia_observada` si está disponible,
   protegiendo contra divisiones por cero en horas nocturnas de baja producción.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.core.config import ENERGY_DISCREPANCY_THRESHOLD_PCT, MODELS_DIR
from src.core.logging_config import get_logger
from src.core.telemetry import track_event

logger = get_logger(__name__)

DEFAULT_TABULAR_MODEL_PATH = MODELS_DIR / "tabular_best_model.joblib"


class TabularInputError(ValueError):
    """Error lanzado cuando las variables tabulares de entrada violan rangos físicos válidos."""


@lru_cache(maxsize=2)
def _load_tabular_bundle(model_path: str = str(DEFAULT_TABULAR_MODEL_PATH)) -> Dict[str, Any]:
    """Carga y cachea el bundle del modelo tabular desde disco."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el modelo tabular entrenado en {path}. "
            "Ejecute primero `python -m src.ml.train_tabular`."
        )
    return joblib.load(path)


def validate_tabular_conditions(
    fecha: str,
    condiciones: Dict[str, Any],
    default_capacity: float = 1100.0,
) -> Dict[str, float]:
    """
    Valida y normaliza las condiciones de entrada para `predecir_generacion`.

    Acepta llaves en español o inglés (ej. `irradiancia` o `IRRADIATION`,
    `temp_amb` o `AMBIENT_TEMPERATURE`, `temp_mod` o `MODULE_TEMPERATURE`,
    `capacidad` o `capacidad_instalada`).
    """
    # Paso 1: Parsear la fecha/hora para extraer `hora` y `dia_anio` si no fueron provistos.
    try:
        dt = datetime.fromisoformat(fecha.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TabularInputError(
            f"Formato de fecha inválido '{fecha}'. Use ISO-8601 (ej. '2020-06-15T12:00:00')."
        ) from exc

    def _pick(*keys: str, default: Optional[float] = None) -> Optional[float]:
        for k in keys:
            if k in condiciones and condiciones[k] is not None:
                return float(condiciones[k])
        return default

    irradiancia = _pick("IRRADIATION", "irradiancia")
    temp_amb = _pick("AMBIENT_TEMPERATURE", "temp_amb", "temperatura_ambiente")
    temp_mod = _pick("MODULE_TEMPERATURE", "temp_mod", "temperatura_modulo")
    capacidad = _pick("capacidad_instalada", "capacidad", default=default_capacity)
    hora = _pick("hora", default=float(dt.hour))
    dia_anio = _pick("dia_anio", default=float(dt.timetuple().tm_yday))

    if irradiancia is None or temp_amb is None or temp_mod is None:
        raise TabularInputError(
            "Faltan variables obligatorias: se requieren irradiancia, "
            "temperatura ambiente y temperatura de módulo."
        )

    # Paso 2: Validación de rangos físicos plausibles en instalaciones fotovoltaicas.
    if irradiancia < 0.0 or irradiancia > 2.5:
        raise TabularInputError(
            f"Irradiancia fuera de rango físico [0.0, 2.5] (normalizada): {irradiancia}"
        )
    if temp_amb < -30.0 or temp_amb > 65.0:
        raise TabularInputError(
            f"Temperatura ambiente fuera de rango físico [-30, 65] °C: {temp_amb}"
        )
    if temp_mod < -30.0 or temp_mod > 105.0:
        raise TabularInputError(
            f"Temperatura de módulo fuera de rango físico [-30, 105] °C: {temp_mod}"
        )
    if capacidad is None or capacidad <= 0.0:
        raise TabularInputError(f"Capacidad instalada debe ser positiva (>0): {capacidad}")
    if hora < 0 or hora > 23:
        raise TabularInputError(f"Hora inválida [0, 23]: {hora}")
    if dia_anio < 1 or dia_anio > 366:
        raise TabularInputError(f"Día del año inválido [1, 366]: {dia_anio}")

    return {
        "IRRADIATION": float(irradiancia),
        "AMBIENT_TEMPERATURE": float(temp_amb),
        "MODULE_TEMPERATURE": float(temp_mod),
        "hora": int(hora),
        "dia_anio": int(dia_anio),
        "capacidad_instalada": float(capacidad),
    }


def predecir_generacion(
    fecha: str,
    condiciones: Dict[str, Any],
    energia_observada: Optional[float] = None,
    panel_id: Optional[str] = None,
    model_path: Path = DEFAULT_TABULAR_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Herramienta principal del Experto 1: predice `energia_kwh_hora` esperada y evalúa
    la discrepancia frente a la generación observada.

    Retorna un diccionario estructurado con:
    - `expected_energy_kwh`: energía esperada predicha (kWh/h).
    - `uncertainty_kwh`: desviación estándar estimada de la predicción.
    - `observed_energy_kwh`: energía real reportada (si fue suministrada).
    - `energy_discrepancy_pct`: déficit porcentual respecto a la esperada (`>= 0`).
    - `signed_deviation_pct`: desviación con signo (`(observada - esperada) / esperada * 100`).
    - `is_anomaly`: `True` si la caída de generación supera `ENERGY_DISCREPANCY_THRESHOLD_PCT`.
    - `model_used`: nombre del modelo tabular activo.
    """
    with track_event(logger, "predict_tabular_energy", fecha=fecha, panel_id=panel_id) as ctx:
        bundle = _load_tabular_bundle(str(model_path))
        capacities: Dict[str, float] = bundle.get("inverter_capacities", {})
        default_cap = float(bundle.get("default_capacity", 1100.0))

        # Si el usuario no especificó capacidad pero sí un SOURCE_KEY conocido, usar su capacidad.
        cond_copy = dict(condiciones)
        if (
            "capacidad_instalada" not in cond_copy
            and "capacidad" not in cond_copy
            and panel_id in capacities
        ):
            cond_copy["capacidad_instalada"] = capacities[panel_id]

        validated = validate_tabular_conditions(fecha, cond_copy, default_capacity=default_cap)
        if energia_observada is not None and energia_observada < 0.0:
            raise TabularInputError(
                f"La energía observada no puede ser negativa: {energia_observada}"
            )

        features: list[str] = bundle["features"]
        pipeline = bundle["pipeline"]
        input_df = pd.DataFrame([{col: validated[col] for col in features}])

        # Paso 3: Predicción puntual acotada físicamente a >= 0.
        raw_pred = float(pipeline.predict(input_df)[0])
        # Si la irradiancia es nula en horario nocturno, acotar a cero para coherencia física.
        if validated["IRRADIATION"] <= 1e-4:
            expected_kwh = 0.0
        else:
            expected_kwh = round(max(0.0, raw_pred), 2)

        # Paso 4: Cuantificación de incertidumbre (residual horario + dispersión entre árboles si RF).
        hora_int = int(validated["hora"])
        residual_map: Dict[int, float] = bundle.get("residual_std_by_hour", {})
        hourly_unc = float(residual_map.get(hora_int, bundle.get("global_residual_std", 25.0)))
        regressor = pipeline.named_steps.get("regressor")
        if hasattr(regressor, "estimators_") and isinstance(regressor.estimators_, list):
            transformed = pipeline.named_steps["preprocessor"].transform(input_df)
            tree_preds = np.array([est.predict(transformed)[0] for est in regressor.estimators_])
            tree_std = float(np.std(tree_preds))
            uncertainty_kwh = round(max(hourly_unc, tree_std), 2)
        else:
            uncertainty_kwh = round(hourly_unc, 2)

        # Paso 5: Cálculo de discrepancia respecto a energía real observada.
        discrepancy_pct: Optional[float] = None
        signed_deviation_pct: Optional[float] = None
        is_anomaly = False

        # Umbral mínimo de energía esperada (5% de capacidad) para evitar falsas alarmas
        # por divisiones casi entre cero al amanecer o atardecer.
        min_active_kwh = max(20.0, 0.05 * validated["capacidad_instalada"])

        if energia_observada is not None:
            if expected_kwh >= min_active_kwh:
                signed_deviation_pct = round(
                    ((float(energia_observada) - expected_kwh) / expected_kwh) * 100.0, 2
                )
                # La discrepancia relevante para inspección mide el déficit de generación.
                deficit_pct = max(
                    0.0, ((expected_kwh - float(energia_observada)) / expected_kwh) * 100.0
                )
                discrepancy_pct = round(deficit_pct, 2)
                is_anomaly = discrepancy_pct >= ENERGY_DISCREPANCY_THRESHOLD_PCT
            else:
                signed_deviation_pct = 0.0
                discrepancy_pct = 0.0
                is_anomaly = False

        result = {
            "panel_id": panel_id or "PLANTA_GENERAL",
            "fecha": fecha,
            "model_used": bundle["model_name"],
            "expected_energy_kwh": expected_kwh,
            "uncertainty_kwh": uncertainty_kwh,
            "observed_energy_kwh": (
                round(float(energia_observada), 2) if energia_observada is not None else None
            ),
            "energy_discrepancy_pct": discrepancy_pct,
            "signed_deviation_pct": signed_deviation_pct,
            "is_anomaly": is_anomaly,
            "threshold_pct": ENERGY_DISCREPANCY_THRESHOLD_PCT,
            "validated_inputs": validated,
        }
        ctx["expected_kwh"] = expected_kwh
        ctx["discrepancy_pct"] = discrepancy_pct
        ctx["is_anomaly"] = is_anomaly
        return result
