"""
Motor de recuperación documental RAG con citas verificables, umbral de evidencia y guardrails.

Paso a paso:
1. Carga perezosa del índice FAISS, la matriz TF-IDF y los metadatos de chunks
   (construyendo el índice automáticamente si aún no existe).
2. Inspecciona la consulta entrante contra patrones de *Prompt Injection* (`guardrails.py`).
3. Calcula una puntuación híbrida verificable (`0.70 * TF-IDF coseno + 0.30 * FAISS denso`)
   para recuperar los `top_k` fragmentos más relevantes con documento y número de página.
4. Aplica un umbral mínimo de similitud (`RAG_MIN_SCORE`): si ningún fragmento supera el
   umbral, declara explícitamente **"Sin respaldo documental suficiente"** para evitar
   alucinaciones o citas inventadas.
5. Expone `consultar_manual_solar(consulta)` como interfaz reutilizable por el agente LangGraph.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.core.config import RAG_MIN_SCORE, VECTOR_STORE_DIR
from src.core.logging_config import get_logger
from src.core.telemetry import track_event
from src.rag.guardrails import inspect_prompt_injection, wrap_passive_rag_context
from src.rag.vector_store import build_vector_store

logger = get_logger(__name__)


@lru_cache(maxsize=2)
def _load_vector_store(store_dir_str: str = str(VECTOR_STORE_DIR)) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    """Carga (o construye si falta) el índice FAISS, bundle TF-IDF/SVD y metadatos de chunks."""
    store_dir = Path(store_dir_str)
    faiss_path = store_dir / "solarguard.faiss"
    vec_path = store_dir / "vectorizer_bundle.joblib"
    chunks_path = store_dir / "chunks.json"

    if not (faiss_path.exists() and vec_path.exists() and chunks_path.exists()):
        logger.warning("Índice vectorial no encontrado; construyendo automáticamente...")
        build_vector_store(store_dir=store_dir)

    index = faiss.read_index(str(faiss_path))
    bundle = joblib.load(vec_path)
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return index, bundle, chunks


def consultar_manual_solar(
    consulta: str,
    top_k: int = 3,
    min_score: float = RAG_MIN_SCORE,
    store_dir: Path = VECTOR_STORE_DIR,
) -> Dict[str, Any]:
    """
    Herramienta principal del Experto 4 (RAG): recupera fragmentos verificables de los
    manuales técnicos fotovoltaicos o declara explícitamente falta de respaldo documental.

    Retorna:
    - `has_evidence`: `True` si al menos un fragmento supera `min_score`.
    - `retrieved_docs`: lista de `{chunk_id, content, source, page, score}`.
    - `citations`: lista formateada `["Documento (p. X)", ...]`.
    - `passive_context`: bloque sanitizado listo para el LLM (datos pasivos).
    - `guardrail_alert`: bandera/mensaje si la consulta o fragmentos contenían inyección.
    - `status_message`: resumen claro incluyendo advertencia cuando no hay evidencia.
    """
    with track_event(logger, "rag_retrieve", top_k=top_k, query_len=len(consulta or "")) as ctx:
        # Paso 1: Defensa contra Prompt Injection en la consulta del usuario.
        query_check = inspect_prompt_injection(consulta, source_label="user_query")
        if not query_check.is_safe:
            ctx["blocked_injection"] = True
            return {
                "query": consulta,
                "has_evidence": False,
                "retrieved_docs": [],
                "citations": [],
                "passive_context": "",
                "guardrail_alert": query_check.reason,
                "top_score": 0.0,
                "status_message": (
                    "Consulta bloqueada parcialmente por políticas de seguridad (Prompt Injection). "
                    "No se ejecutarán instrucciones fuera del protocolo técnico autorizado."
                ),
            }

        clean_query = query_check.sanitized_text.strip()
        if not clean_query:
            return {
                "query": consulta,
                "has_evidence": False,
                "retrieved_docs": [],
                "citations": [],
                "passive_context": "",
                "guardrail_alert": None,
                "top_score": 0.0,
                "status_message": "Consulta vacía; no hay respaldo documental.",
            }

        index, bundle, chunks = _load_vector_store(str(store_dir))
        vectorizer = bundle["vectorizer"]
        svd = bundle["svd"]
        tfidf_matrix = bundle["tfidf_matrix"]

        # Paso 2: Calcular similitud dispersa TF-IDF (precisión léxica) y densa FAISS (semántica).
        q_tfidf = vectorizer.transform([clean_query])
        sparse_scores = cosine_similarity(q_tfidf, tfidf_matrix).ravel()

        q_dense = svd.transform(q_tfidf).astype(np.float32)
        q_dense = normalize(q_dense, norm="l2").astype(np.float32)
        dense_sims, dense_idxs = index.search(q_dense, len(chunks))

        dense_scores = np.zeros(len(chunks), dtype=np.float32)
        for rank_i, doc_idx in enumerate(dense_idxs[0]):
            if 0 <= doc_idx < len(chunks):
                dense_scores[doc_idx] = max(0.0, float(dense_sims[0][rank_i]))

        # Combinación ponderada: TF-IDF domina para evitar falsos positivos fuera de dominio.
        # Además medimos la cobertura léxica de términos de la consulta sobre el vocabulario
        # del corpus: si menos del 35% de los términos clave de la consulta aparecen en el corpus
        # (ej. consulta sobre "Huawei SUN2000 Modbus" donde solo coincide "inversor"), se penaliza
        # para evitar alucinar respaldo documental con una sola palabra genérica.
        analyzer = vectorizer.build_analyzer()
        query_tokens = [tok for tok in analyzer(clean_query) if " " not in tok]
        vocab = vectorizer.vocabulary_
        if query_tokens:
            covered = sum(1 for tok in query_tokens if tok in vocab)
            vocab_coverage = covered / len(query_tokens)
        else:
            vocab_coverage = 0.0

        raw_hybrid = np.where(
            sparse_scores > 0.03,
            0.70 * sparse_scores + 0.30 * dense_scores,
            0.15 * dense_scores,
        )
        # Factor de cobertura: si < 35% de los términos de la pregunta existen en el corpus,
        # atenúa el puntaje proporcionalmente para activar la abstención segura.
        coverage_factor = 1.0 if vocab_coverage >= 0.35 else (vocab_coverage / 0.35) * 0.4
        hybrid_scores = raw_hybrid * coverage_factor

        # Paso 3: Ordenar y filtrar por umbral mínimo de evidencia `min_score`.
        ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        selected_docs: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            score = round(float(hybrid_scores[int(idx)]), 4)
            if score < min_score:
                continue
            chunk = chunks[int(idx)]
            selected_docs.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "source": chunk["source"],
                    "page": int(chunk["page"]),
                    "score": score,
                    "content": chunk["content"],
                }
            )

        top_score = round(float(np.max(hybrid_scores)), 4) if len(hybrid_scores) > 0 else 0.0
        has_evidence = len(selected_docs) > 0

        passive_context, doc_injection = wrap_passive_rag_context(selected_docs)
        citations = [f"{d['source']} (p. {d['page']})" for d in selected_docs]

        if not has_evidence:
            status_msg = (
                f"Sin respaldo documental suficiente en el corpus (score máximo={top_score:.3f} "
                f"< umbral={min_score:.2f}). No se inventan procedimientos ni citas."
            )
            logger.warning(f"[RAG] {status_msg} query='{clean_query[:60]}'")
        else:
            status_msg = f"Recuperados {len(selected_docs)} fragmentos verificables (top_score={top_score:.3f})."

        ctx["has_evidence"] = has_evidence
        ctx["top_score"] = top_score
        ctx["n_docs"] = len(selected_docs)

        return {
            "query": consulta,
            "has_evidence": has_evidence,
            "retrieved_docs": selected_docs,
            "citations": citations,
            "passive_context": passive_context,
            "guardrail_alert": (
                "Contenido anómalo neutralizado en fragmentos." if doc_injection else None
            ),
            "top_score": top_score,
            "status_message": status_msg,
        }
