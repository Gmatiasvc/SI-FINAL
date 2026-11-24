# SolarGuard AI — Sistema Inteligente Híbrido para Inspección Fotovoltaica (Grupo 08 · UNT)

> **Alcance implementado:** Sistema completo **SolarGuard AI** (ML tabular anti-fuga, Visión RGB, Detección Espacial YOLOv8n, Termografía IR condicional, RAG con FAISS y guardrails, Orquestación Agéntica MoE en LangGraph con Human-in-the-Loop, Telemetría diaria coloreada, Informe en LaTeX, Suite `pytest` + `AppTest` e **Interfaz Web Multipágina en Streamlit**).

## Arquitectura Mixture of Experts (MoE) en LangGraph

```text
validar_entradas ──(error)──> manejar_error
       │
       ▼
predecir_energia (Experto 1: GradientBoostingRegressor)
       │
       ├──[discrepancia < 15% sin imagen]──> redactar_reporte_normal
       │
       └──[discrepancia >= 15% o imagen]──> detectar_y_clasificar_rgb (Experto 2A YOLOv8n + 2B MobileNetV3)
                                                    │
                  ┌──[vista no evaluable / <0.60]───┼──[evaluable + IR]──> clasificar_termografia (Experto 3)
                  ▼                                 │                              │
      detener_por_incertidumbre                     └──────────┬───────────────────┘
                                                               ▼
                                                        consultar_rag (Experto 4: FAISS + Guardrails)
                                                               │
                                                               ▼
                                                  generar_propuesta_revision (Pydantic)
                                                               │
                                                [interrupt_before: HITL]
                                                               ▼
                                                        validar_tecnico
                                                               │
                                                     [aprobado por humano]
                                                               ▼
                                                    simular_orden_revision
```

## Estructura del Repositorio

- `src/core/`: Configuración global (`config.py`), logging dual coloreado en `stdout` y archivo diario `logs/solarguard_YYYY-MM-DD.log` (`logging_config.py`), y telemetría (`telemetry.py`).
- `src/data/`: Preparación anti-fuga temporal (`tabular_prep.py`), deduplicación perceptual pHash (`vision_prep.py`), submuestreo balanceado IR (`thermal_prep.py`) y splits YOLO (`yolo_prep.py`).
- `src/ml/`: Entrenamiento comparativo (`train_tabular.py`) e inferencia con validación física (`predict_tabular.py`).
- `src/vision/`: Entrenamientos secuenciales (`train_rgb.py`, `train_thermal.py`, `train_yolo.py`) e inferencia unificada (`inference.py`).
- `src/rag/`: Generación del corpus PDF (`corpus_builder.py`), índice FAISS (`vector_store.py`), recuperación verificable (`retriever.py`), defensas anti-Prompt Injection (`guardrails.py`) y benchmark de 20 consultas (`evaluate_rag.py`).
- `src/agents/`: Estado tipado (`state.py`), esquemas Pydantic (`schemas.py`), 4 herramientas `@tool` (`tools.py`), grafo LangGraph (`graph.py`) y ejecutor CLI de escenarios (`runner.py`).
- `tests/`: Pruebas automatizadas con `pytest`.
- `report/`: Informe técnico-académico en LaTeX (`informe_solarguard_ai.tex` y `informe_solarguard_ai.pdf`) con gráficas en `report/figures/`.

## Guía Rápida de Ejecución (Secuencial en CPU)

```bash
# 1. Activar entorno virtual
source .venv/bin/activate

# 2. Entrenar módulos secuencialmente (uno a la vez para no saturar CPU/RAM)
python -m src.ml.train_tabular
python -m src.vision.train_rgb
python -m src.vision.train_thermal
python -m src.vision.train_yolo

# 3. Construir corpus RAG y ejecutar benchmark de 20 preguntas
python -m src.rag.evaluate_rag

# 4. Ejecutar escenarios integrales del agente LangGraph (MoE + HITL) por CLI
python -m src.agents.runner

# 5. Lanzar la interfaz web multipágina en Streamlit
streamlit run streamlit_app.py

# 6. Ejecutar suite completa de pruebas automatizadas (Backend + UI AppTest)
pytest -v
```

## Políticas de Seguridad y Guardrails
1. **No Data Leakage:** Partición temporal consecutiva en ML tabular; agrupación por *Perceptual Hash* (`imagehash.phash`) antes del split en visión RGB; generación sintética post-split.
2. **Control Humano Obligatorio:** El grafo pausa antes de `validar_tecnico` (`interrupt_before`) y exige aprobación explícita antes de emitir órdenes simuladas.
3. **Prohibición Operativa:** Nunca se emiten comandos de desconexión eléctrica automatizada ni se infieren fallos internos sin captura térmica o medición adecuada.
4. **Registro y Observabilidad:** Todos los eventos emiten telemetría coloreada en `stdout` y registro limpio en `logs/solarguard_YYYY-MM-DD.log`.
