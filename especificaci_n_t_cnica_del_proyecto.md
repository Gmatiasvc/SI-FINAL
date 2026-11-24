# Especificación Técnica: SolarGuard AI (Grupo 08 - UNT)

## 1. Resumen Ejecutivo y Arquitectura MoE Agéntica
**SolarGuard AI** es un sistema inteligente híbrido diseñado para recomendar inspecciones técnicas en instalaciones fotovoltaicas mediante el cruce de pronósticos de generación eléctrica, inspección visual (RGB y termografía opcional) y documentación técnica indexada (RAG).

### Patrón Mixture of Experts (MoE) Orquestado
El sistema implementa un **MoE funcional a nivel de orquestación en LangGraph** (enrutamiento disperso y condicional), donde **LangGraph actúa como la red compuerta (*Router / Gating Network*)** que invoca modelos especializados independientes según la evidencia disponible:
* **Experto 1 — Predictor de Línea Base (Tabular ML):** Estima `energia_kwh_hora` esperada según variables meteorológicas y calcula la discrepancia frente a la generación real observada.
* **Experto 2A — Detector Espacial (YOLOv8n / YOLOv11n):** Localiza y recorta paneles individuales en tomas amplias (métrica: `mAP@50`).
* **Experto 2B — Clasificador Visual RGB (EfficientNet-B0 / MobileNetV3):** Clasifica cada recorte en las 4 clases obligatorias (`panel limpio`, `suciedad visible`, `grieta visible`, `vista no evaluable`).
* **Experto 3 — Inspector Térmico Condicional (MobileNetV3-Small IR):** Se activa **únicamente** si el usuario adjunta una captura termográfica; clasifica anomalías térmicas (`Normal térmico`, `Hotspot / Celda crítica`, `Fallo de diodo / Bypass`).
* **Experto 4 — Asistente Documental (RAG + LLM):** Recupera protocolos de mantenimiento y seguridad desde manuales técnicos con citas verificables (documento y página).

---

## 2. Datasets Seleccionados y Estrategia Anti-Fuga (*No Data Leakage*)

### 2.1 Módulo Tabular (Regresión)
* **Dataset:** *Solar Power Generation Data* (`anikannal` en Kaggle).
* **Variables de entrada:** `IRRADIATION`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `hora`, `dia_anio`, `capacidad_instalada` (derivada por `SOURCE_KEY`).
* **Variable objetivo:** `energia_kwh_hora` (agregación horaria de `AC_POWER` / `DC_POWER`).
* **Partición estricta:** División **temporal** por bloques de días consecutivos (ej. Días 1–24 Train, Días 25–29 Val, Días 30–34 Test). Prohibido el muestreo aleatorio. Uso obligatorio de `sklearn.pipeline.Pipeline` y `ColumnTransformer`.
* **Modelos:**
  * *Baseline:* `LinearRegression` (solo irradiancia y capacidad).
  * *Candidatos:* `RandomForestRegressor` y `GradientBoostingRegressor` (o `HistGradientBoostingRegressor`).
* **Métricas:** MAE (kWh), RMSE y Error Normalizado por Capacidad (nMAE / nRMSE).

### 2.2 Módulo de Visión RGB (Clasificación 4 Clases)
* **Dataset:** *Solar Panel Images (Clean and Faulty Images)* (`pythonafroz` en Kaggle).
* **Mapeo obligatorio de etiquetas:**
  1. `panel limpio` $\leftarrow$ `Clean`
  2. `suciedad visible` $\leftarrow$ `Dusty` + `Bird-drop`
  3. `grieta visible` $\leftarrow$ `Physical-Damage`
  4. `vista no evaluable` $\leftarrow$ `Snow-Covered` + subconjunto sintético controlado de desenfoque intenso (*Gaussian Blur*), sobreexposición (*Glare*) y oclusión extrema.
* **Partición estricta:** Deduplicación previa mediante *Perceptual Hashing* (`imagehash`) para agrupar imágenes similares o aumentadas en el mismo *split* (70% Train / 15% Val / 15% Test).
* **Métricas:** `Macro-F1` y `Recall` en las clases de daño (`suciedad visible` y `grieta visible`).

### 2.3 Extensión 1: Detección de Paneles (YOLO)
* **Dataset:** *Solar Panel Object Detection Dataset* (Roboflow Universe — `project-bcvkh/solar-panel-yn6y2`, Licencia CC BY 4.0).
* **Métrica:** `mAP@50`.

### 2.4 Extensión 2: Termografía Opcional (IR)
* **Dataset:** *Infrared Solar Modules* (Raptor Maps — GitHub oficial).
* **Submuestra:** ~2,000 imágenes balanceadas agrupadas en 3 clases: `Normal térmico` (`No-Anomaly`), `Hotspot / Celda crítica` (`Hot-Spot`, `Cell`, `Cell-Multi`), `Fallo de diodo` (`Diode`, `Diode-Multi`).

---

## 3. Arquitectura del Agente en LangGraph

### 3.1 Estado Tipado (`SolarGuardState`)
```python
from typing import TypedDict, List, Optional, Dict, Any

class SolarGuardState(TypedDict):
    # Entradas
    panel_id: str
    telemetry: Dict[str, float]          # irradiancia, temp_amb, temp_mod, capacidad, energia_observada
    rgb_image_path: Optional[str]
    thermal_image_path: Optional[str]
    user_query: Optional[str]
    
    # Resultados de Expertos
    expected_energy_kwh: Optional[float]
    energy_discrepancy_pct: Optional[float]
    is_anomaly: bool
    detected_panels_crops: List[str]
    rgb_classification: Optional[Dict[str, Any]]      # {label, confidence, all_probs}
    thermal_classification: Optional[Dict[str, Any]]  # {label, confidence}
    
    # RAG y Salida Estructurada
    retrieved_docs: List[Dict[str, Any]]              # {content, source, page, score}
    structured_report: Optional[Dict[str, Any]]       # Validado por Pydantic
    
    # Control, Trazabilidad y Human-in-the-Loop
    uncertainty_flag: bool
    human_approval_required: bool
    human_approved: Optional[bool]
    trace_log: List[str]
    errors: List[str]
```

### 3.2 Flujo de Nodos y Rutas Condicionales
1. `validar_entradas`: Verifica rangos físicos (ej. irradiancia $\ge 0$) e integridad de archivos de imagen. Si falla $\rightarrow$ `manejar_error`.
2. `predecir_energia`: Invoca `predecir_generacion()`. Calcula desviación porcentual respecto a `energia_observada`.
3. **Router Condicional 1 (`evaluar_discrepancia`):**
   * Si `discrepancia < 15%` y el usuario no forzó inspección visual (Escenario 1) $\rightarrow$ `redactar_reporte_normal`.
   * Si `discrepancia >= 15%` o hay imagen adjunta $\rightarrow$ `detectar_y_clasificar_rgb`.
4. `detectar_y_clasificar_rgb`: Ejecuta YOLO (recorte) + EfficientNet/MobileNetV3 (clasificación).
5. **Router Condicional 2 (`evaluar_confianza_y_termica`):**
   * Si `label == "vista no evaluable"` o `confidence < 0.60` $\rightarrow$ activa `uncertainty_flag = True` y detiene el diagnóstico solicitando nueva captura.
   * Si `thermal_image_path` está presente $\rightarrow$ `clasificar_termografia` $\rightarrow$ `consultar_rag`.
   * Si no hay imagen térmica $\rightarrow$ `consultar_rag`.
6. `consultar_rag`: Busca en índice vectorial (Chroma/FAISS) usando la combinación `[nivel_discrepancia + condicion_visual + condicion_termica]`.
7. `generar_propuesta_revision`: LLM genera salida estructurada bajo esquema Pydantic `{panel, energia_esperada, desviacion, condicion_visual, pasos_revision, citas}`.
8. `validar_tecnico` (**Human-in-the-Loop**): Punto de interrupción (`interrupt_before`) en LangGraph. Streamlit muestra el botón de aprobación/rechazo antes de ejecutar `simular_orden_revision`.

---

## 4. Guardrails, Seguridad y Requisitos de Feria
* **Restricción Ética/Operativa:** Prohibido emitir comandos de desconexión eléctrica o diagnóstico de fallos internos sin medición térmica/eléctrica.
* **Defensa contra Prompt Injection:** El prompt del sistema instruye explícitamente que los fragmentos recuperados del RAG son *datos pasivos*, nunca instrucciones ejecutables.
* **Evaluación RAG:** Benchmark obligatorio de **20 preguntas** (contestables, sin respaldo documental y adversariales) comparando respuestas *Sin RAG* vs *Con RAG* (`Recall@k`, fidelidad y exactitud de citas).
* **Modo Offline / Demo:** Caché local de resultados y soporte para LLM local (Ollama) para garantizar operatividad sin internet en la feria.