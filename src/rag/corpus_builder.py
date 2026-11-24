"""
Constructor del corpus documental fotovoltaico verificado en formato PDF paginado.

Paso a paso:
1. Genera 4 manuales y guías técnicas estructuradas en PDF dentro de `data/corpus/`
   usando `reportlab`, con encabezados claros, numeración de páginas explícita y contenido
   técnico basado en estándares reales de O&M fotovoltaica (IEC 62446-3, NREL O&M).
2. Permite que el motor RAG (`vector_store.py`) extraiga texto real página por página con
   `pypdf`, garantizando citas trazables `(documento, página)` y separando claramente
   las normas operativas (ej. prohibición de desconexión automatizada).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from src.core.config import CORPUS_DIR
from src.core.logging_config import get_logger
from src.core.telemetry import track_event

logger = get_logger(__name__)

# Definición estructurada del corpus técnico oficial por documento y página.
CORPUS_DOCUMENTS: Dict[str, Dict[str, object]] = {
    "Manual_Mantenimiento_Inversor_FV_2026.pdf": {
        "title": "Manual Técnico de Operación y Mantenimiento de Inversores Fotovoltaicos (Ed. 2026)",
        "pages": [
            (
                "Sección 1.1 — Diagnóstico de Baja Generación Eléctrica vs. Irradiancia",
                "Ante una caída de potencia activa (AC_POWER) en inversores de cadena, el operador debe "
                "comparar la energía observada (kWh/h) contra la línea base esperada ajustada por "
                "irradiancia solar y temperatura de módulo. Una diferencia inferior al 15% dentro de la "
                "franja diurna puede deberse a nubosidad transitoria o variación térmica normal. Cuando la "
                "discrepancia energética esperada supera el umbral del 15%, se debe priorizar una "
                "inspección visual del campo solar antes de intervenir el inversor.",
            ),
            (
                "Sección 1.2 — Controles Manuales en Campo ante Alerta de Inversor",
                "Cuando la baja generación coincide con suciedad visible o sombreado parcial, los "
                "controles manuales recomendados incluyen: (1) verificar lecturas de tensión DC (Voc) y "
                "corriente por string en la pantalla local del inversor; (2) inspeccionar visualmente "
                "fusibles de string y conectores MC4 sin abrir circuitos bajo carga; (3) registrar la "
                "hora exacta e irradiancia ambiental en la bitácora de mantenimiento.",
            ),
            (
                "Sección 1.3 — Restricción Crítica de Desconexión Eléctrica",
                "PROHIBICIÓN OPERATIVA: Ningún sistema de software o recomendación automatizada puede "
                "ejecutar comandos remotos de desconexión eléctrica (breaker trip o apertura de seccionador "
                "DC bajo carga). Toda maniobra de aislamiento o bloqueo/etiquetado (LOTO) debe ser "
                "autorizada y ejecutada presencialmente por un técnico electricista habilitado tras "
                "confirmar ausencia de corriente con pinza amperimétrica certificada.",
            ),
        ],
    },
    "Guia_Inspeccion_Visual_Paneles_RGB.pdf": {
        "title": "Guía de Campo para Inspección Visual RGB de Módulos Fotovoltaicos",
        "pages": [
            (
                "Sección 2.1 — Clasificación: Panel Limpio vs. Suciedad Visible (Polvo y Aves)",
                "Un módulo en condición 'panel limpio' presenta superficie de vidrio uniforme, sin depósitos "
                "opacos ni obstrucciones que reduzcan la transmitancia óptica. La categoría 'suciedad visible' "
                "abarca acumulación de polvo mineral, hollín y excrementos de aves ('bird-drop'). Los "
                "excrementos focalizados generan sombreado duro de alta densidad capaz de activar diodos "
                "bypass o inducir calentamiento localizado si no se limpian oportunamente.",
            ),
            (
                "Sección 2.2 — Protocolo Seguro de Limpieza de Paneles con Suciedad Visible",
                "Para tratar 'suciedad visible', programar limpieza en horas de baja irradiancia (amanecer "
                "o atardecer) para evitar choque térmico en el vidrio templado. Emplear agua "
                "desmineralizada o blanda (< 50 ppm TDS) a baja presión (< 35 bar) junto con cepillos de "
                "cerdas suaves no abrasivas o escobillas de goma limpia. Está prohibido usar detergentes "
                "ácidos, solventes orgánicos o hidrolavadoras de alta presión a corta distancia.",
            ),
            (
                "Sección 2.3 — Protocolo ante Grieta Visible o Daño Físico del Vidrio",
                "Ante la detección de 'grieta visible' (vidrio templado estrellado, fractura lineal o "
                "delaminación severa), existe riesgo de ingreso de humedad, pérdida de aislamiento y falla "
                "a tierra. Se debe: (1) acordonar y señalar el módulo sin aplicar agua ni presión mecánica; "
                "(2) prohibir la limpieza húmeda hasta evaluar la resistencia de aislamiento (Megger); "
                "(3) inspeccionar visualmente la lámina posterior (backsheet) y planificar reemplazo.",
            ),
            (
                "Sección 2.4 — Criterios de Calidad Fotográfica: Vista No Evaluable",
                "Una fotografía se clasifica como 'vista no evaluable' cuando presenta cobertura total o "
                "parcial por nieve, desenfoque intenso por movimiento, reflejo solar especular (glare) que "
                "satura el vidrio o oclusión mayor al 50% del encuadre. Ante una vista no evaluable o "
                "confianza del clasificador inferior a 0.60, se debe detener el diagnóstico automático y "
                "solicitar al operador una nueva captura perpendicular (ángulo 15-30 grados para evitar "
                "reflejos) con iluminación difusa adecuada.",
            ),
        ],
    },
    "Protocolo_Termografia_IR_IEC62446.pdf": {
        "title": "Protocolo de Termografía Infrarroja Condicional en Plantas FV (Ref. IEC 62446-3)",
        "pages": [
            (
                "Sección 3.1 — Condiciones de Validez y Alcance de la Termografía",
                "La evaluación termográfica infrarroja (IR) es un módulo condicional que solo se ejecuta "
                "cuando se adjuntan capturas obtenidas con cámara térmica calibrada (emisividad típica del "
                "vidrio ~0.85). Para diagnósticos válidos en campo se recomienda una irradiancia en plano "
                "de arreglo >= 600 W/m2 y velocidad de viento baja. Nunca deben diagnosticarse fallos "
                "internos de celdas o diodos únicamente a partir de fotografías RGB sin evidencia térmica "
                "o medición eléctrica directa.",
            ),
            (
                "Sección 3.2 — Interpretación: Normal Térmico y Hotspot / Celda Crítica",
                "En condición 'Normal térmico', la distribución de temperatura en el módulo es homogénea "
                "(variación < 5 °C respecto a módulos vecinos). Un 'Hotspot / Celda crítica' se manifiesta "
                "como un punto caliente puntual o múltiple con diferencial térmico Delta-T >= 10 °C a "
                "20 °C sobre celdas adyacentes. Causas frecuentes: excremento puntual, microfisura de "
                "celda o soldadura degradada. Pasos de revisión: verificar ausencia de sombra externa y "
                "ejecutar prueba de curva I-V en el string afectado.",
            ),
            (
                "Sección 3.3 — Interpretación: Fallo de Diodo / Bypass",
                "Un 'Fallo de diodo / Bypass' se observa típicamente como una banda longitudinal completa "
                "(un tercio o mitad del módulo) a temperatura uniformemente elevada respecto al resto del "
                "panel. Indica que el diodo de bypass en la caja de conexiones está conduciendo en "
                "permanencia o en cortocircuito/circuito abierto. Pasos de revisión: inspeccionar caja de "
                "conexiones con cámara IR y verificar caída de tensión y continuidad de diodos con "
                "multímetro tras aislamiento manual seguro del string.",
            ),
        ],
    },
    "Normativa_Seguridad_O_y_M_Fotovoltaica.pdf": {
        "title": "Normativa de Seguridad Eléctrica y Control Humano en O&M Fotovoltaica",
        "pages": [
            (
                "Sección 4.1 — Seguridad Eléctrica y Equipos de Protección Personal (EPP)",
                "Todo trabajo de revisión física sobre arreglos FV energizados requiere calzado aislante, "
                "guantes dieléctricos clase 0 (hasta 1,000 V DC) y protección ocular UV. Los paneles "
                "generan tensión continua peligrosa ante cualquier iluminación ambiental incluso si el "
                "inversor se encuentra apagado.",
            ),
            (
                "Sección 4.2 — Flujo de Aprobación Humana (Human-in-the-Loop)",
                "Las recomendaciones generadas por SolarGuard AI constituyen órdenes sugeridas de "
                "inspección técnica basadas en discrepancias estadísticas y evidencia visual. La emisión "
                "definitiva de la orden de trabajo requiere validación explícita del técnico o jefe de "
                "planta, quien debe corroborar que las citas documentales y el nivel de incertidumbre "
                "justifican la movilización de cuadrilla.",
            ),
        ],
    },
}


def build_corpus_pdfs(corpus_dir: Path = CORPUS_DIR, overwrite: bool = True) -> List[Path]:
    """
    Genera los PDFs paginados del corpus técnico dentro de `data/corpus/`.
    Retorna la lista de rutas creadas.
    """
    with track_event(logger, "build_corpus_pdfs", corpus_dir=str(corpus_dir)) as ctx:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "SolarDocTitle",
            parent=styles["Heading1"],
            fontSize=15,
            spaceAfter=14,
        )
        sec_style = ParagraphStyle(
            "SolarSecTitle",
            parent=styles["Heading2"],
            fontSize=12,
            spaceAfter=8,
        )
        body_style = ParagraphStyle(
            "SolarBody",
            parent=styles["BodyText"],
            fontSize=10.5,
            leading=15,
        )

        created_files: List[Path] = []
        for filename, spec in CORPUS_DOCUMENTS.items():
            pdf_path = corpus_dir / filename
            if pdf_path.exists() and not overwrite:
                created_files.append(pdf_path)
                continue

            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            story: List[object] = [
                Paragraph(str(spec["title"]), title_style),
                Spacer(1, 8),
            ]
            pages: List[tuple[str, str]] = spec["pages"]  # type: ignore[assignment]
            for idx, (sec_title, sec_body) in enumerate(pages):
                story.append(Paragraph(sec_title, sec_style))
                story.append(Paragraph(sec_body, body_style))
                if idx < len(pages) - 1:
                    story.append(PageBreak())

            doc.build(story)
            created_files.append(pdf_path)
            logger.info(f"Documento PDF generado: {pdf_path.name} ({len(pages)} páginas)")

        ctx["num_docs"] = len(created_files)
        return created_files


if __name__ == "__main__":
    build_corpus_pdfs()
