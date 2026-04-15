import json
import logging
import sqlite3
import warnings
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", message="CropBox missing from /Page, defaulting to MediaBox")
logging.getLogger("pypdf").setLevel(logging.ERROR)


class RAGEngine:
    """
    Retrieval engine backed by lightweight SQLite persistence.
    Embeddings are stored as binary blobs and loaded in memory for fast search.
    """

    def __init__(
        self,
        files_dir: str = "files",
        chunk_size: int = 300,
        overlap: int = 50,
        db_dir: str = "vector_store",
        db_name: str = "rag_index.sqlite3",
        embedding_model: str = "all-MiniLM-L6-v2",
        add_batch_size: int = 128,
    ):
        self.files_dir = Path(files_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.db_dir = Path(db_dir)
        self.db_name = db_name
        self.embedding_model = embedding_model
        self.add_batch_size = add_batch_size

        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / self.db_name
        self.model = SentenceTransformer(self.embedding_model)
        self._lock = RLock()

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_db()

        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self._doc_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_documents(self, force_rebuild: bool = False) -> int:
        with self._lock:
            return self._load_documents_locked(force_rebuild=force_rebuild)

    def sync_index_if_needed(self) -> bool:
        """
        Check whether files changed since last index build.
        Rebuild only when needed. Returns True if rebuild/clear happened.
        """
        with self._lock:
            all_files = self._list_source_files()
            if not all_files:
                had_index = self._db_has_chunks_locked() or bool(self.chunks)
                if had_index:
                    self._clear_index_locked()
                    print(f"[RAG] No PDF or TXT files found in '{self.files_dir}'. Cleared index.")
                    return True
                return False

            manifest = self._build_manifest(all_files)
            if self._can_reuse_existing_index_locked(manifest):
                if not self.chunks and self._db_has_chunks_locked():
                    self._hydrate_from_sqlite_locked()
                return False

            print("[RAG] Detected changes in files/. Rebuilding index...")
            self._load_documents_locked(force_rebuild=True)
            return True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            if not self.chunks and self._db_has_chunks_locked():
                self._hydrate_from_sqlite_locked()

            if not self.chunks or self.embeddings is None:
                return []

            n_results = min(top_k, len(self.chunks))
            if n_results <= 0:
                return []

            query_vec = self.model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
            sims = np.dot(self.embeddings, query_vec) / np.maximum(norms, 1e-10)

            top_indices = np.argsort(sims)[::-1][:n_results]
            rows: List[Dict[str, Any]] = []
            for idx in top_indices:
                chunk = self.chunks[idx]
                rows.append(
                    {
                        "text": chunk["text"],
                        "document": chunk["document"],
                        "page": chunk["page"],
                        "chunk_id": chunk["chunk_id"],
                        "score": float(sims[idx]),
                    }
                )
            return rows

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._lock:
            if not self._doc_counts and self._db_has_chunks_locked():
                self._hydrate_from_sqlite_locked()
            return [{"name": name, "chunks": count} for name, count in self._doc_counts.items()]

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------------
    # Indexing internals
    # ------------------------------------------------------------------

    def _load_documents_locked(self, force_rebuild: bool = False) -> int:
        self.chunks = []
        self._doc_counts = {}
        self.embeddings = None

        all_files = self._list_source_files()
        if not all_files:
            self._clear_index_locked()
            print(f"[RAG] No PDF or TXT files found in '{self.files_dir}'")
            return 0

        manifest = self._build_manifest(all_files)
        if not force_rebuild and self._can_reuse_existing_index_locked(manifest):
            self._hydrate_from_sqlite_locked()
            print(f"[RAG] Using persisted SQLite index ({len(self.chunks)} chunks).")
            return len(self.chunks)

        pdf_files = [path for path in all_files if path.suffix.lower() == ".pdf"]
        txt_files = [path for path in all_files if path.suffix.lower() == ".txt"]

        for pdf_path in pdf_files:
            print(f"[RAG] Processing: {pdf_path.name}")
            self._process_pdf(pdf_path)

        for txt_path in txt_files:
            print(f"[RAG] Processing: {txt_path.name}")
            self._process_txt(txt_path)

        if not self.chunks:
            self._clear_index_locked()
            print("[RAG] No chunks were extracted from available files.")
            return 0

        print(f"[RAG] Encoding and indexing {len(self.chunks)} chunks in SQLite...")
        texts = [item["text"] for item in self.chunks]
        self.embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        self._persist_index_locked(manifest)
        print("[RAG] SQLite index ready.")
        return len(self.chunks)

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                page INTEGER NOT NULL,
                chunk_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def _persist_index_locked(self, manifest: Dict[str, Any]) -> None:
        rows = []
        for idx, chunk in enumerate(self.chunks):
            rows.append(
                (
                    chunk["id"],
                    chunk["document"],
                    chunk["page"],
                    chunk["chunk_id"],
                    chunk["text"],
                    self.embeddings[idx].tobytes(),
                )
            )

        cur = self.conn.cursor()
        cur.execute("DELETE FROM chunks")
        cur.executemany(
            """
            INSERT INTO chunks (id, document, page, chunk_id, text, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        cur.execute(
            """
            INSERT INTO metadata (key, value) VALUES ('manifest_json', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (json.dumps(manifest, sort_keys=True),),
        )
        self.conn.commit()

    def _hydrate_from_sqlite_locked(self) -> None:
        self.chunks = []
        self._doc_counts = {}
        self.embeddings = None

        rows = self.conn.execute(
            "SELECT id, document, page, chunk_id, text, embedding FROM chunks ORDER BY rowid"
        ).fetchall()
        if not rows:
            return

        vectors: List[np.ndarray] = []
        for row in rows:
            document = str(row["document"])
            page = int(row["page"])
            chunk_id = int(row["chunk_id"])
            text = str(row["text"])
            vector = np.frombuffer(row["embedding"], dtype=np.float32)

            self.chunks.append(
                {
                    "id": row["id"],
                    "document": document,
                    "page": page,
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )
            self._doc_counts[document] = self._doc_counts.get(document, 0) + 1
            vectors.append(vector)

        self.embeddings = np.vstack(vectors).astype(np.float32)

    def _clear_index_locked(self) -> None:
        self.conn.execute("DELETE FROM chunks")
        self.conn.execute("DELETE FROM metadata WHERE key='manifest_json'")
        self.conn.commit()
        self.chunks = []
        self._doc_counts = {}
        self.embeddings = None

    def _db_has_chunks_locked(self) -> bool:
        row = self.conn.execute("SELECT COUNT(1) AS count FROM chunks").fetchone()
        return bool(row and int(row["count"]) > 0)

    def _can_reuse_existing_index_locked(self, manifest: Dict[str, Any]) -> bool:
        if not self._db_has_chunks_locked():
            return False

        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key='manifest_json'"
        ).fetchone()
        if not row:
            return False

        try:
            saved_manifest = json.loads(row["value"])
        except Exception:
            return False

        return saved_manifest == manifest

    def _list_source_files(self) -> List[Path]:
        files = list(self.files_dir.glob("*.pdf")) + list(self.files_dir.glob("*.txt"))
        return sorted(files, key=lambda p: p.name.lower())

    def _build_manifest(self, paths: List[Path]) -> Dict[str, Any]:
        files_state: List[Dict[str, Any]] = []
        for path in paths:
            stat = path.stat()
            files_state.append(
                {
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "embedding_model": self.embedding_model,
            "files": files_state,
        }

    # ------------------------------------------------------------------
    # Document processing helpers
    # ------------------------------------------------------------------

    def _process_pdf(self, pdf_path: Path) -> None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text or not text.strip():
                        continue
                    for chunk_text in self._split_into_chunks(text):
                        self._append_chunk(
                            text=chunk_text,
                            document=pdf_path.name,
                            page=page_num,
                        )
        except Exception as exc:
            print(f"[RAG] Warning - could not read '{pdf_path.name}': {exc}")

    def _process_txt(self, txt_path: Path) -> None:
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                return
            for chunk_text in self._split_into_chunks(text):
                self._append_chunk(
                    text=chunk_text,
                    document=txt_path.name,
                    page=1,
                )
        except Exception as exc:
            print(f"[RAG] Warning - could not read '{txt_path.name}': {exc}")

    def _append_chunk(self, text: str, document: str, page: int) -> None:
        chunk_id = len(self.chunks)
        self.chunks.append(
            {
                "id": f"{document}:{page}:{chunk_id}",
                "text": text,
                "document": document,
                "page": page,
                "chunk_id": chunk_id,
            }
        )
        self._doc_counts[document] = self._doc_counts.get(document, 0) + 1

    def _split_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        if not words:
            return []

        step = max(self.chunk_size - self.overlap, 1)
        chunks: List[str] = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + self.chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
            i += step
        return chunks
