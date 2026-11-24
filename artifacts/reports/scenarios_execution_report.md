# Reporte de Escenarios Ejecutados — SolarGuard AI

## escenario_1_normal (147.87 ms)
- **Panel:** INV-1BY6WEcLGh8j5v7
- **Energía esperada:** 969.66 kWh/h (±33.17 kWh/h)
- **Desviación:** -3.06% (déficit 3.06% < umbral 15.0%)
- **Condición visual:** No requerida (generación dentro de línea base normal)
- **Estado orden HITL:** NO_REQUERIDA
- **Citas RAG:** ['Manual_Mantenimiento_Inversor_FV_2026.pdf (p. 1)', 'Manual_Mantenimiento_Inversor_FV_2026.pdf (p. 2)']
- **Trazas LangGraph:**
  - `[validar_entradas] Entradas válidas`
  - `[predecir_energia] esperada=969.66 kWh/h (±33.17), observada=940.0, discrepancia=3.06%, anomalia=False`
  - `[redactar_reporte_normal] Operación normal (desviación=3.06%, rag_evidence=True)`

## escenario_2_vision_rgb_suciedad (564.66 ms)
- **Panel:** STRING-04-PANEL-12
- **Energía esperada:** 969.66 kWh/h (±33.17 kWh/h)
- **Desviación:** -2.03% (déficit 2.03%)
- **Condición visual:** suciedad visible (confianza 1.00)
- **Estado orden HITL:** EMITIDA (ORD-20261124-CD9C5D)
- **Citas RAG:** ['Guia_Inspeccion_Visual_Paneles_RGB.pdf (p. 2)', 'Guia_Inspeccion_Visual_Paneles_RGB.pdf (p. 1)']
- **Trazas LangGraph:**
  - `[validar_entradas] Entradas válidas`
  - `[predecir_energia] esperada=969.66 kWh/h (±33.17), observada=950.0, discrepancia=2.03%, anomalia=False`
  - `[detectar_y_clasificar_rgb] label='suciedad visible', conf=1.000, es_evaluable=True, crops=1`
  - `[consultar_rag] evidence=True, citas=['Guia_Inspeccion_Visual_Paneles_RGB.pdf (p. 2)', 'Guia_Inspeccion_Visual_Paneles_RGB.pdf (p. 1)']`
  - `[generar_propuesta_revision] Propuesta estructurada Pydantic lista; esperando validación humana (HITL).`
  - `[validar_tecnico] [HITL] Técnico APROBÓ la propuesta. Nota='Limpieza programada para mañana 06:00 AM.'`
  - `[simular_orden_revision] Orden simulada emitida: ORD-20261124-CD9C5D (prioridad=MEDIA)`

## escenario_3_integral_ml_rgb_ir (177.87 ms)
- **Panel:** STRING-07-PANEL-03
- **Energía esperada:** 969.66 kWh/h (±33.17 kWh/h)
- **Desviación:** -39.15% (déficit 39.15%)
- **Condición visual:** grieta visible (confianza 1.00)
- **Condición térmica:** Normal térmico (confianza 0.47)
- **Estado orden HITL:** EMITIDA (ORD-20261124-850E3B)
- **Citas RAG:** ['Manual_Mantenimiento_Inversor_FV_2026.pdf (p. 2)', 'Manual_Mantenimiento_Inversor_FV_2026.pdf (p. 1)']
- **Trazas LangGraph:**
  - `[validar_entradas] Entradas válidas`
  - `[predecir_energia] esperada=969.66 kWh/h (±33.17), observada=590.0, discrepancia=39.15%, anomalia=True`
  - `[detectar_y_clasificar_rgb] label='grieta visible', conf=0.999, es_evaluable=True, crops=1`
  - `[clasificar_termografia] thermal_label='Normal térmico', conf=0.467`
  - `[consultar_rag] evidence=True, citas=['Manual_Mantenimiento_Inversor_FV_2026.pdf (p. 2)', 'Manual_Mantenimiento_Inversor_FV_2026.pdf (p. 1)']`
  - `[generar_propuesta_revision] Propuesta estructurada Pydantic lista; esperando validación humana (HITL).`
  - `[validar_tecnico] [HITL] Técnico APROBÓ la propuesta. Nota='Confirmada grieta + hotspot; cuadrilla despachada con Megger.'`
  - `[simular_orden_revision] Orden simulada emitida: ORD-20261124-850E3B (prioridad=ALTA)`

## caso_dificil_vista_no_evaluable (134.35 ms)
- **Panel:** STRING-02-PANEL-09
- **Energía esperada:** 969.66 kWh/h (±33.17 kWh/h)
- **Desviación:** Déficit 36.06%
- **Condición visual:** vista no evaluable (confianza=0.62)
- **Estado orden HITL:** DETENIDA_POR_INCERTIDUMBRE
- **Citas RAG:** ['Guia_Inspeccion_Visual_Paneles_RGB.pdf (p. 4)']
- **Trazas LangGraph:**
  - `[validar_entradas] Entradas válidas`
  - `[predecir_energia] esperada=969.66 kWh/h (±33.17), observada=620.0, discrepancia=36.06%, anomalia=True`
  - `[detectar_y_clasificar_rgb] label='vista no evaluable', conf=0.622, es_evaluable=False, crops=1`
  - `[detener_por_incertidumbre] Imagen clasificada como 'vista no evaluable' (confianza=0.62 < umbral=0.60 o nieve/oclusión). Se detiene el diagnóstico automático y se solicita nueva captura.`

## caso_error_entrada_invalida (4.77 ms)
- **Panel:** INV-ERR-01
- **Energía esperada:** No calculada por error en entrada
- **Desviación:** No disponible
- **Condición visual:** No evaluada
- **Estado orden HITL:** ERROR_ENTRADA
- **Citas RAG:** []
- **Trazas LangGraph:**
  - `[validar_entradas] Errores detectados (1)`
  - `[manejar_error] Flujo abortado: ['Error en telemetría tabular: Irradiancia fuera de rango físico [0.0, 2.5] (normalizada): -0.5']`

## caso_sin_respaldo_y_rechazo_hitl (137.23 ms)
- **Panel:** STRING-09-PANEL-01
- **Energía esperada:** 969.66 kWh/h (±33.17 kWh/h)
- **Desviación:** -37.09% (déficit 37.09%)
- **Condición visual:** suciedad visible (confianza 1.00)
- **Estado orden HITL:** RECHAZADA_POR_TECNICO
- **Citas RAG:** []
- **Trazas LangGraph:**
  - `[validar_entradas] Entradas válidas`
  - `[predecir_energia] esperada=969.66 kWh/h (±33.17), observada=610.0, discrepancia=37.09%, anomalia=True`
  - `[detectar_y_clasificar_rgb] label='suciedad visible', conf=1.000, es_evaluable=True, crops=1`
  - `[consultar_rag] evidence=False, citas=[]`
  - `[generar_propuesta_revision] Propuesta estructurada Pydantic lista; esperando validación humana (HITL).`
  - `[validar_tecnico] [HITL] Técnico RECHAZÓ o canceló la orden. Nota='Rechazada por falta de manual específico de firmware.'`
