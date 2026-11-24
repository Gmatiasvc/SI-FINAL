"""Pruebas unitarias del sistema de logging diario, colores ANSI en stdout y telemetría."""

from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path

from src.core.logging_config import (
    ColoredConsoleFormatter,
    DailyFileHandler,
    PlainFileFormatter,
    get_daily_log_path,
    get_logger,
)
from src.core.telemetry import TelemetryRecorder, track_event


def test_colored_formatter_includes_ansi_and_plain_strips_ansi() -> None:
    record = logging.LogRecord(
        name="solarguard.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=10,
        msg="[GUARDRAIL] Prueba de color \033[32mverde\033[0m",
        args=(),
        exc_info=None,
    )
    colored = ColoredConsoleFormatter(use_color=True).format(record)
    plain = PlainFileFormatter().format(record)

    assert "\033[33m" in colored  # Amarillo para WARNING
    assert "\033[35m" in colored  # Magenta para [GUARDRAIL]
    assert "\033[" not in plain   # Archivo .log debe quedar limpio sin ANSI


def test_daily_log_file_created_and_telemetry_recorded(tmp_path: Path) -> None:
    handler = DailyFileHandler(logs_dir=tmp_path, prefix="solarguard")
    handler.setFormatter(PlainFileFormatter())
    test_logger = logging.getLogger("solarguard.unit_tmp")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(handler)
    test_logger.propagate = False

    recorder = TelemetryRecorder()
    with track_event(test_logger, "unit_telemetry_check", recorder=recorder, sample=1):
        test_logger.info("Evento de prueba diario")

    handler.flush()
    handler.close()
    test_logger.removeHandler(handler)

    today_str = datetime.now().strftime("%Y-%m-%d")
    expected_file = tmp_path / f"solarguard_{today_str}.log"
    assert expected_file.exists()
    content = expected_file.read_text(encoding="utf-8")
    assert "[TELEMETRY] START event=unit_telemetry_check" in content
    assert "[TELEMETRY] OK    event=unit_telemetry_check" in content
    assert "\033[" not in content
    assert len(recorder.events) == 1
    assert recorder.events[0].status == "OK"


def test_default_logger_and_daily_path() -> None:
    log = get_logger("tests.logging")
    log.info("Verificando logger principal con salida diaria")
    assert get_daily_log_path().exists()
