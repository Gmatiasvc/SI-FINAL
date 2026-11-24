"""
Módulo de telemetría y registro estructurado de eventos para SolarGuard AI.

Paso a paso:
1. Lee el consumo de memoria RSS del proceso actual desde `/proc/self/status` (Linux)
   o `resource.getrusage` como respaldo sin dependencias externas pesadas.
2. Provee el context manager `track_event(logger, event_name, **metadata)` que registra
   el inicio y la finalización de cualquier operación (entrenamiento, inferencia, RAG o nodo),
   midiendo latencia exacta en milisegundos y reportando errores si ocurren.
3. Provee `TelemetryRecorder` para acumular trazas estructuradas por nodo/herramienta durante
   la ejecución del grafo LangGraph y exportarlas en JSON o reportes.
"""

from __future__ import annotations

import json
import logging
import resource
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


def get_process_memory_mb() -> float:
    """
    Obtiene la memoria residente actual (VmRSS) del proceso en Megabytes (MB).
    En Linux lee `/proc/self/status` para conocer la memoria física usada en tiempo real;
    si no está disponible, utiliza `resource.getrusage`.
    """
    proc_status = Path("/proc/self/status")
    if proc_status.exists():
        try:
            for line in proc_status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return round(kb / 1024.0, 2)
        except Exception:
            pass
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss_kb / 1024.0, 2)


@dataclass
class TelemetryEvent:
    """Representación tipada de un evento de telemetría del sistema."""

    event_name: str
    status: str
    duration_ms: float
    memory_mb: float
    timestamp_utc: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetryRecorder:
    """
    Acumulador en memoria de eventos de telemetría durante una sesión o flujo del agente.
    Permite exportar una línea de tiempo auditada a JSON o mostrar resúmenes en logs.
    """

    def __init__(self) -> None:
        self.events: List[TelemetryEvent] = []

    def record(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def to_list(self) -> List[Dict[str, Any]]:
        return [ev.to_dict() for ev in self.events]

    def export_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_list(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


@contextmanager
def track_event(
    logger: logging.Logger,
    event_name: str,
    recorder: Optional[TelemetryRecorder] = None,
    **metadata: Any,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager para instrumentar funciones, etapas de entrenamiento o nodos LangGraph.

    Uso:
        with track_event(logger, "train_tabular_rf", n_rows=1000) as ctx:
            ...
            ctx["mae"] = 12.34

    Registra en el logger:
    - Inicio del evento con metadatos de entrada.
    - Duración en milisegundos, memoria RSS actual (MB) y metadatos actualizados al finalizar.
    - En caso de excepción, emite nivel ERROR con traza completa y propaga el error.
    """
    start_time = time.perf_counter()
    mem_before = get_process_memory_mb()
    meta_str = json.dumps(metadata, ensure_ascii=False, default=str) if metadata else "{}"
    logger.info(f"[TELEMETRY] START event={event_name} mem_mb={mem_before:.2f} meta={meta_str}")

    context_data: Dict[str, Any] = dict(metadata)
    try:
        yield context_data
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        mem_after = get_process_memory_mb()
        logger.error(
            f"[TELEMETRY] ERROR event={event_name} duration_ms={elapsed_ms:.2f} "
            f"mem_mb={mem_after:.2f} error={type(exc).__name__}: {exc}",
            exc_info=True,
        )
        if recorder is not None:
            recorder.record(
                TelemetryEvent(
                    event_name=event_name,
                    status="ERROR",
                    duration_ms=round(elapsed_ms, 2),
                    memory_mb=mem_after,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    metadata=context_data,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        mem_after = get_process_memory_mb()
        out_str = (
            json.dumps(context_data, ensure_ascii=False, default=str) if context_data else "{}"
        )
        logger.info(
            f"[TELEMETRY] OK    event={event_name} duration_ms={elapsed_ms:.2f} "
            f"mem_mb={mem_after:.2f} details={out_str}"
        )
        if recorder is not None:
            recorder.record(
                TelemetryEvent(
                    event_name=event_name,
                    status="OK",
                    duration_ms=round(elapsed_ms, 2),
                    memory_mb=mem_after,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    metadata=context_data,
                )
            )


def log_metric(logger: logging.Logger, module: str, metrics: Dict[str, Any]) -> None:
    """Emite una línea estandarizada de métricas verificables hacia stdout y `.log` diario."""
    payload = json.dumps(metrics, ensure_ascii=False, default=str)
    logger.info(f"[METRIC] module={module} metrics={payload}")
