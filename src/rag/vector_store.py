"""
Extracción, chunking e indexación vectorial persistente (FAISS + TF-IDF/LSA) para el RAG.

Paso a paso:
1. Lee todos los archivos PDF en `data/corpus/` (generándolos vía `build_corpus_pdfs()` si
   el directorio está vacío) mediante `pypdf.PdfReader`, preservando número de página exacto.
2. Limpia el texto extraído y segmenta cada página con `RecursiveCharacterTextSplitter` de
   LangChain (`chunk_size=550`, `chunk_overlap=100`), manteniendo trazabilidad completa:
   `chunk_id`, `doc_id`, `source`, `page` y `content`.
3. Construye representaciones vectoriales L2-normalizadas combinando:
   - `TfidfVectorizer` (unigramas y bigramas) para precisión en términos técnicos FV.
   - `TruncatedSVD` (embeddings densos LSA) indexados en `faiss.IndexFlatIP` persistido en disco.
4. Guarda en `artifacts/vector_store/`: `solarguard.faiss`, `vectorizer_bundle.joblib` y
   `chunks.json`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import faiss
import joblib
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.core.config import CORPUS_DIR, RANDOM_SEED, VECTOR_STORE_DIR
from src.core.logging_config import get_logger
from src.core.telemetry import track_event
from src.rag.corpus_builder import build_corpus_pdfs

logger = get_logger(__name__)

# Stopwords mínimas en español para no distorsionar la similitud coseno en consultas cortas.
SPANISH_STOPWORDS = [
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "y", "o", "en", "con", "por", "para", "que", "se", "su", "sus", "es", "son",
    "como", "qué", "cual", "cuál", "cuáles", "ante", "sobre", "entre", "más",
]


def _clean_text(text: str) -> str:
    """Normaliza espacios y saltos de línea conservando puntuación técnica."""
    return re.sub(r"\s+", " ", text or "").strip()


def extract_and_chunk_corpus(
    corpus_dir: Path = CORPUS_DIR,
    chunk_size: int = 550,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Extrae texto página por página de cada PDF y genera chunks trazables.
    """
    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info("Corpus vacío; generando PDFs técnicos base...")
        build_corpus_pdfs(corpus_dir=corpus_dir)
        pdf_files = sorted(corpus_dir.glob("*.pdf"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Dict[str, Any]] = []
    for doc_idx, pdf_path in enumerate(pdf_files, start=1):
        reader = PdfReader(str(pdf_path))
        doc_id = f"DOC_{doc_idx:02d}"
        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            clean = _clean_text(raw_text)
            if not clean:
                continue
            pieces = splitter.split_text(clean)
            for piece_idx, piece in enumerate(pieces, start=1):
                chunks.append(
                    {
                        "chunk_id": f"{doc_id}_P{page_num:02d}_C{piece_idx:02d}",
                        "doc_id": doc_id,
                        "source": pdf_path.name,
                        "page": page_num,
                        "content": piece,
                    }
                )
    logger.info(f"Extraídos {len(chunks)} fragmentos desde {len(pdf_files)} documentos PDF.")
    return chunks


def build_vector_store(
    corpus_dir: Path = CORPUS_DIR,
    store_dir: Path = VECTOR_STORE_DIR,
) -> Dict[str, Any]:
    """
    Construye y persiste el índice vectorial FAISS + vectorizador TF-IDF/LSA en disco.
    """
    with track_event(logger, "build_rag_vector_store", corpus_dir=str(corpus_dir)) as ctx:
        store_dir.mkdir(parents=True, exist_ok=True)
        chunks = extract_and_chunk_corpus(corpus_dir=corpus_dir)
        if not chunks:
            raise RuntimeError("No se extrajeron fragmentos del corpus documental.")

        texts = [c["content"] for c in chunks]

        # Paso 1: Matriz dispersa TF-IDF (n-gramas 1-2) normalizada L2.
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words=SPANISH_STOPWORDS,
            max_features=4096,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Paso 2: Proyección densa LSA normalizada L2 para indexación en FAISS IndexFlatIP.
        n_comp = min(64, max(2, min(tfidf_matrix.shape) - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_SEED)
        dense_vectors = svd.fit_transform(tfidf_matrix).astype(np.float32)
        dense_vectors = normalize(dense_vectors, norm="l2").astype(np.float32)

        index = faiss.IndexFlatIP(dense_vectors.shape[1])
        index.add(dense_vectors)

        faiss_path = store_dir / "solarguard.faiss"
        vec_path = store_dir / "vectorizer_bundle.joblib"
        chunks_path = store_dir / "chunks.json"

        faiss.write_index(index, str(faiss_path))
        joblib.dump(
            {
                "vectorizer": vectorizer,
                "svd": svd,
                "tfidf_matrix": tfidf_matrix,
            },
            vec_path,
        )
        chunks_path.write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        ctx["num_chunks"] = len(chunks)
        ctx["faiss_dim"] = dense_vectors.shape[1]
        logger.info(
            f"Índice vectorial FAISS persistido en {store_dir} "
            f"(chunks={len(chunks)}, dim={dense_vectors.shape[1]})."
        )
        return {
            "num_chunks": len(chunks),
            "faiss_dim": int(dense_vectors.shape[1]),
            "store_dir": str(store_dir),
        }


if __name__ == "__main__":
    build_vector_store()
