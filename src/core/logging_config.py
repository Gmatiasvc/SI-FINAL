"""
Sistema centralizado de logging para SolarGuard AI.

Características implementadas paso a paso:
1. Salida a `stdout` con formato legible y códigos de color ANSI diferenciados por nivel
   (DEBUG=cyan, INFO=verde, WARNING=amarillo, ERROR=rojo, CRITICAL=rojo negrita).
2. Salida simultánea a un archivo de registro diario ubicado en `logs/solarguard_YYYY-MM-DD.log`
   en texto plano UTF-8 (eliminando cualquier código ANSI antes de escribir en disco).
3. Rotación automática por cambio de día mediante un FileHandler dinámico que verifica
   la fecha en cada evento emitido.
4. Configuración idempotente sobre el logger raíz `solarguard` para evitar duplicación de mensajes.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.core.config import LOG_LEVEL, LOGS_DIR

# Expresión regular para limpiar secuencias de escape ANSI antes de guardar en archivo .log.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

# Diccionario de colores ANSI estándar para terminal/stdout.
ANSI_COLORS: dict[str, str] = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[90m",
    "CYAN": "\033[36m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "RED": "\033[31m",
    "BOLD_RED": "\033[1;31m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
}

LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: ANSI_COLORS["CYAN"],
    logging.INFO: ANSI_COLORS["GREEN"],
    logging.WARNING: ANSI_COLORS["YELLOW"],
    logging.ERROR: ANSI_COLORS["RED"],
    logging.CRITICAL: ANSI_COLORS["BOLD_RED"],
}


class ColoredConsoleFormatter(logging.Formatter):
    """
    Formateador personalizado que aplica colores ANSI a la salida por `stdout`.

    Paso a paso:
    - Colorea la marca de tiempo en gris tenue.
    - Colorea la etiqueta del nivel de severidad según `LEVEL_COLORS`.
    - Colorea el nombre del módulo en azul para distinguir rápidamente el origen del evento.
    - Resalta etiquetas operativas como `[TELEMETRY]`, `[NODE]` o `[GUARDRAIL]` en magenta.
    """

    def __init__(self, use_color: bool = True) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # Paso 1: Generar timestamp legible con milisegundos.
        timestamp = self.formatTime(record, self.datefmt)
        ms = int(record.msecs)
        full_ts = f"{timestamp}.{ms:03d}"

        # Paso 2: Obtener mensaje base (incluyendo traceback si hubo excepción).
        message = record.getMessage()
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message = f"{message}\n{record.exc_text}"

        # Paso 3: Si no se usan colores, devolver texto limpio.
        if not self.use_color:
            return f"{full_ts} | {record.levelname:<8} | {record.name} | {message}"

        # Paso 4: Aplicar colores ANSI a cada componente visual.
        c_dim = ANSI_COLORS["DIM"]
        c_lvl = LEVEL_COLORS.get(record.levelno, ANSI_COLORS["RESET"])
        c_mod = ANSI_COLORS["BLUE"]
        c_tag = ANSI_COLORS["MAGENTA"]
        c_rst = ANSI_COLORS["RESET"]

        # Resaltar etiquetas clave de telemetría o trazas LangGraph dentro del mensaje.
        highlighted_msg = message
        for tag in ("[TELEMETRY]", "[NODE]", "[TOOL]", "[GUARDRAIL]", "[HITL]", "[METRIC]"):
            if tag in highlighted_msg:
                highlighted_msg = highlighted_msg.replace(tag, f"{c_tag}{tag}{c_rst}")

        return (
            f"{c_dim}{full_ts}{c_rst} | "
            f"{c_lvl}{record.levelname:<8}{c_rst} | "
            f"{c_mod}{record.name}{c_rst} | "
            f"{highlighted_msg}"
        )


class PlainFileFormatter(logging.Formatter):
    """
    Formateador para archivos `.log` diarios que garantiza texto plano limpio sin secuencias ANSI.
    """

    def __init__(self) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        ms = int(record.msecs)
        full_ts = f"{timestamp}.{ms:03d}"
        message = record.getMessage()
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message = f"{message}\n{record.exc_text}"

        # Limpiar cualquier secuencia ANSI que pudiera venir dentro del mensaje.
        clean_msg = _ANSI_ESCAPE_RE.sub("", message)
        return f"{full_ts} | {record.levelname:<8} | {record.name} | {clean_msg}"


class DailyFileHandler(logging.Handler):
    """
    Handler personalizado que escribe registros en un archivo nombrado por fecha:
    `logs/solarguard_YYYY-MM-DD.log`.

    Si el proceso cruza la medianoche durante un entrenamiento largo, cambia
    automáticamente al archivo del nuevo día sin perder eventos.
    """

    def __init__(self, logs_dir: Path = LOGS_DIR, prefix: str = "solarguard") -> None:
        super().__init__()
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self._current_date: str = ""
        self._file_handler: Optional[logging.FileHandler] = None
        self._ensure_handler_for_today()

    @property
    def current_log_path(self) -> Path:
        """Retorna la ruta del archivo `.log` correspondiente al día en curso."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.logs_dir / f"{self.prefix}_{today}.log"

    def _ensure_handler_for_today(self) -> None:
        """Abre o rota el FileHandler interno si cambió el día calendario."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date or self._file_handler is None:
            if self._file_handler is not None:
                self._file_handler.close()
            self._current_date = today
            log_path = self.logs_dir / f"{self.prefix}_{today}.log"
            self._file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            if self.formatter is not None:
                self._file_handler.setFormatter(self.formatter)

    def setFormatter(self, fmt: Optional[logging.Formatter]) -> None:
        super().setFormatter(fmt)
        if self._file_handler is not None and fmt is not None:
            self._file_handler.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_handler_for_today()
            if self._file_handler is not None:
                self._file_handler.emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._file_handler is not None:
            self._file_handler.close()
            self._file_handler = None
        super().close()


def get_daily_log_path(logs_dir: Path = LOGS_DIR) -> Path:
    """Devuelve la ruta esperada del archivo de log diario actual."""
    today = datetime.now().strftime("%Y-%m-%d")
    return Path(logs_dir) / f"solarguard_{today}.log"


def setup_logging(
    level: Optional[str] = None,
    logs_dir: Path = LOGS_DIR,
    force_color: Optional[bool] = None,
) -> logging.Logger:
    """
    Configura de manera idempotente el logger raíz `solarguard`.

    Paso a paso:
    1. Determina el nivel de log (DEBUG, INFO, WARNING, ERROR).
    2. Verifica si el logger `solarguard` ya fue configurado para no duplicar handlers.
    3. Adjunta un `StreamHandler(sys.stdout)` con `ColoredConsoleFormatter` (con colores ANSI).
    4. Adjunta un `DailyFileHandler` con `PlainFileFormatter` hacia `logs/solarguard_YYYY-MM-DD.log`.
    """
    root_pkg_logger = logging.getLogger("solarguard")
    target_level = getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO)
    root_pkg_logger.setLevel(target_level)
    root_pkg_logger.propagate = False

    # Evitar añadir handlers duplicados en múltiples llamadas a get_logger().
    if getattr(root_pkg_logger, "_solarguard_configured", False):
        return root_pkg_logger

    # Por defecto activamos color ANSI en stdout porque el usuario lo solicitó explícitamente,
    # permitiendo desactivarlo sólo si SOLARGUARD_COLOR_LOGS=0.
    if force_color is None:
        use_color = os.getenv("SOLARGUARD_COLOR_LOGS", "1") != "0"
    else:
        use_color = force_color

    # Handler 1: Consola stdout con colores ANSI.
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(target_level)
    console_handler.setFormatter(ColoredConsoleFormatter(use_color=use_color))

    # Handler 2: Archivo diario en disco sin códigos ANSI.
    file_handler = DailyFileHandler(logs_dir=logs_dir)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(PlainFileFormatter())

    root_pkg_logger.addHandler(console_handler)
    root_pkg_logger.addHandler(file_handler)
    root_pkg_logger._solarguard_configured = True  # type: ignore[attr-defined]
    return root_pkg_logger


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Obtiene un logger hijo bajo la jerarquía `solarguard.*`.

    Ejemplo: `get_logger(__name__)` con `src.ml.train_tabular` produce `solarguard.src.ml.train_tabular`,
    heredando automáticamente el stdout coloreado y el archivo `.log` diario.
    """
    setup_logging(level=level)
    clean_name = name if name.startswith("solarguard") else f"solarguard.{name}"
    return logging.getLogger(clean_name)


def log_section(logger: logging.Logger, title: str) -> None:
    """Emite un separador visual limpio en logs y stdout para delimitar fases del pipeline."""
    bar = "=" * 72
    logger.info(bar)
    logger.info(f"FASE: {title}")
    logger.info(bar)
