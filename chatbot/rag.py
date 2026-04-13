import os
import numpy as np
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.
    Loads PDFs from a directory, chunks them, embeds with sentence-transformers,
    and retrieves the most relevant chunks for a given query via cosine similarity.
    """

    def __init__(self, files_dir: str = "files", chunk_size: int = 300, overlap: int = 50):
        self.files_dir = Path(files_dir)
        self.chunk_size = chunk_size   # words per chunk
        self.overlap = overlap          # words overlap between chunks
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_documents(self) -> int:
        """
        Scan files_dir for PDF files, extract text, chunk it, and build
        the embedding matrix.  Returns the total number of chunks created.
        """
        self.chunks = []
        pdf_files = sorted(self.files_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"[RAG] No PDF files found in '{self.files_dir}'")
            return 0

        for pdf_path in pdf_files:
            print(f"[RAG] Processing: {pdf_path.name}")
            self._process_pdf(pdf_path)

        if self.chunks:
            print(f"[RAG] Encoding {len(self.chunks)} chunks …")
            texts = [c["text"] for c in self.chunks]
            self.embeddings = self.model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            print("[RAG] Embeddings ready.")

        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Return the top_k most relevant chunks for *query*.
        Each result dict contains: text, document, page, chunk_id, score.
        """
        if not self.chunks or self.embeddings is None:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)[0]

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        sims = np.dot(self.embeddings, query_vec) / np.maximum(norms, 1e-10)

        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            entry = self.chunks[idx].copy()
            entry["score"] = float(sims[idx])
            results.append(entry)

        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return per-document chunk counts."""
        doc_map: Dict[str, int] = {}
        for chunk in self.chunks:
            doc_map[chunk["document"]] = doc_map.get(chunk["document"], 0) + 1
        return [{"name": k, "chunks": v} for k, v in doc_map.items()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_pdf(self, pdf_path: Path) -> None:
        """Extract text page-by-page and append chunks to self.chunks."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text or not text.strip():
                        continue
                    for chunk_text in self._split_into_chunks(text):
                        self.chunks.append(
                            {
                                "text": chunk_text,
                                "document": pdf_path.name,
                                "page": page_num,
                                "chunk_id": len(self.chunks),
                            }
                        )
        except Exception as exc:
            print(f"[RAG] Warning – could not read '{pdf_path.name}': {exc}")

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split *text* into overlapping word-based chunks.
        Returns a list of non-empty chunk strings.
        """
        words = text.split()
        chunks: List[str] = []
        step = max(self.chunk_size - self.overlap, 1)
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += step
        return chunks
