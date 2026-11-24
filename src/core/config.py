"""
Configuración centralizada de rutas, semillas, hiperparámetros de seguridad,
límites de CPU/RAM y umbrales de decisión para SolarGuard AI.

Paso a paso:
1. Define la raíz del proyecto de forma relativa al archivo actual para asegurar portabilidad.
2. Carga variables de entorno desde `.env` si existe, manteniendo valores seguros por defecto.
3. Expone rutas absolutas a datasets, artefactos, logs, corpus documental y figuras del informe.
4. Provee una función `configure_cpu_limits()` que restringe los hilos de BLAS/PyTorch/OpenMP
   para evitar saturar la CPU/RAM al entrenar redes neuronales de forma secuencial.
"""

from __future__ import annotations

import os
from pathlib import Path

# Paso 1: Intentar cargar variables de entorno desde un archivo `.env` local (si existe).
# Se usa try/except para que el módulo no falle si python-dotenv aún no está disponible.
try:
    from dotenv import load_dotenv

    _PROJECT_CANDIDATE = Path(__file__).resolve().parents[2]
    _ENV_FILE = _PROJECT_CANDIDATE / ".env"
    if _ENV_FILE.exists():
        load_dotenv(dotenv_path=_ENV_FILE)
except ImportError:
    pass

# Paso 2: Definir rutas base del repositorio.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DATA_DIR: Path = PROJECT_ROOT / "data"
CORPUS_DIR: Path = DATA_DIR / "corpus"
REPORT_DIR: Path = PROJECT_ROOT / "report"
FIGURES_DIR: Path = REPORT_DIR / "figures"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

# Paso 3: Rutas específicas de los 4 conjuntos de datos proporcionados.
TABULAR_DATASET_DIR: Path = DATASETS_DIR / "Solar Power Generation Data"
RGB_DATASET_DIR: Path = DATASETS_DIR / "Solar Panel Images Clean and Faulty Images"
YOLO_DATASET_DIR: Path = DATASETS_DIR / "mixed_1 Computer Vision Dataset"
IR_DATASET_DIR: Path = DATASETS_DIR / "InfraredSolarModules"

# Paso 4: Subdirectorios de artefactos reproducibles generados por los pipelines.
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
SPLITS_DIR: Path = ARTIFACTS_DIR / "splits"
ORDERS_DIR: Path = ARTIFACTS_DIR / "orders"
CACHE_DIR: Path = ARTIFACTS_DIR / "cache"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
VECTOR_STORE_DIR: Path = ARTIFACTS_DIR / "vector_store"

# Asegurar que los directorios críticos de salida existan al importar la configuración.
for _directory in (
    ARTIFACTS_DIR,
    MODELS_DIR,
    METRICS_DIR,
    SPLITS_DIR,
    ORDERS_DIR,
    CACHE_DIR,
    REPORTS_DIR,
    VECTOR_STORE_DIR,
    LOGS_DIR,
    DATA_DIR,
    CORPUS_DIR,
    REPORT_DIR,
    FIGURES_DIR,
    CONFIGS_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)

# Paso 5: Semilla global de reproducibilidad y límites conservadores de cómputo en CPU.
RANDOM_SEED: int = int(os.getenv("SOLARGUARD_SEED", "42"))
MAX_CPU_THREADS: int = int(os.getenv("SOLARGUARD_MAX_CPU_THREADS", "4"))

# Paso 6: Umbrales de decisión del sistema MoE (Mixture of Experts) y RAG.
# Si la desviación entre energía esperada y observada es >= 15%, se dispara inspección visual.
ENERGY_DISCREPANCY_THRESHOLD_PCT: float = float(
    os.getenv("SOLARGUARD_DISCREPANCY_PCT", "15.0")
)
# Si la confianza del clasificador RGB cae por debajo de 0.60, se marca como "vista no evaluable".
VISION_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("SOLARGUARD_VISION_CONF_THRESHOLD", "0.60")
)
# Umbral mínimo de confianza del detector YOLO para aceptar recortes de paneles.
YOLO_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("SOLARGUARD_YOLO_CONF_THRESHOLD", "0.25")
)
# Puntuación mínima de similitud en RAG para considerar que existe respaldo documental suficiente.
RAG_MIN_SCORE: float = float(os.getenv("SOLARGUARD_RAG_MIN_SCORE", "0.20"))

# Paso 7: Etiquetas canónicas obligatorias para Visión RGB (4 clases) y Termografía IR (3 clases).
RGB_CLASSES: list[str] = [
    "panel limpio",
    "suciedad visible",
    "grieta visible",
    "vista no evaluable",
]

# Mapeo explícito de carpetas crudas del dataset RGB hacia las 4 clases obligatorias.
# Nota ética/técnica: se excluye 'Electrical-damage' porque los fallos eléctricos internos no deben
# inferirse únicamente desde fotografías RGB sin medición eléctrica o termográfica adecuada.
RGB_FOLDER_MAPPING: dict[str, str] = {
    "Clean": "panel limpio",
    "Dusty": "suciedad visible",
    "Bird-drop": "suciedad visible",
    "Physical-Damage": "grieta visible",
    "Snow-Covered": "vista no evaluable",
}

THERMAL_CLASSES: list[str] = [
    "Normal térmico",
    "Hotspot / Celda crítica",
    "Fallo de diodo / Bypass",
]

# Mapeo de etiquetas crudas del dataset Raptor Maps InfraredSolarModules hacia las 3 clases IR.
THERMAL_CLASS_MAPPING: dict[str, str] = {
    "No-Anomaly": "Normal térmico",
    "Hot-Spot": "Hotspot / Celda crítica",
    "Hot-Spot-Multi": "Hotspot / Celda crítica",
    "Cell": "Hotspot / Celda crítica",
    "Cell-Multi": "Hotspot / Celda crítica",
    "Diode": "Fallo de diodo / Bypass",
    "Diode-Multi": "Fallo de diodo / Bypass",
}

# Paso 8: Configuración del proveedor LLM y modo offline/demo para operación sin internet.
LLM_PROVIDER: str = os.getenv("SOLARGUARD_LLM_PROVIDER", "offline").lower()
LLM_MODEL: str = os.getenv("SOLARGUARD_LLM_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOG_LEVEL: str = os.getenv("SOLARGUARD_LOG_LEVEL", "INFO").upper()


def configure_cpu_limits(max_threads: int = MAX_CPU_THREADS) -> int:
    """
    Restringe el número de hilos utilizados por OpenMP, MKL, OpenBLAS y PyTorch
    para prevenir congelamientos del sistema durante entrenamientos secuenciales en CPU.

    Debe invocarse al inicio de los scripts de entrenamiento antes de instanciar
    modelos pesados o DataLoaders.
    """
    threads_str = str(max(1, int(max_threads)))
    for env_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[env_var] = threads_str

    # Si PyTorch está instalado, ajustar también sus hilos intra-op e inter-op.
    try:
        import torch

        torch.set_num_threads(int(threads_str))
        # Evitar fallo si interop threads ya fue inicializado previamente.
        try:
            torch.set_num_interop_threads(max(1, int(threads_str) // 2))
        except RuntimeError:
            pass
    except ImportError:
        pass

    return int(threads_str)
