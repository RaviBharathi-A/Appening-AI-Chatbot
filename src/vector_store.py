"""Persistent SQLite-backed TF-IDF vector search for the local document collection."""

import json
import math
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from langchain_core.documents import Document

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "according",
    "be",
    "by",
    "can",
    "do",
    "does",
    "document",
    "ebook",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "outlined",
    "the",
    "their",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "mentioned",
}


class LocalVectorStore:
    """Store document term vectors and retrieve them by TF-IDF cosine similarity."""

    def __init__(
        self,
        path: str | Path,
        collection: str,
    ) -> None:
        self.path = Path(path).expanduser()
        self.collection = collection
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            schema = connection.execute(
                "PRAGMA table_info(vectors)"
            ).fetchall()
            if not schema:
                self._create_table(connection)
            else:
                columns = {row[1] for row in schema}
                legacy_embedding_required = any(
                    row[1] == "embedding" and row[3] for row in schema
                )
                if legacy_embedding_required:
                    self._migrate_legacy_table(connection, "term_frequencies" in columns)
                elif "term_frequencies" not in columns:
                    connection.execute(
                        "ALTER TABLE vectors ADD COLUMN term_frequencies TEXT"
                    )

    @staticmethod
    def _create_table(connection: sqlite3.Connection, name: str = "vectors") -> None:
        connection.execute(
            f"""
            CREATE TABLE {name} (
                collection TEXT NOT NULL,
                vector_id TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                term_frequencies TEXT,
                PRIMARY KEY (collection, vector_id)
            )
            """
        )

    def _migrate_legacy_table(
        self, connection: sqlite3.Connection, has_term_frequencies: bool
    ) -> None:
        term_column = "term_frequencies" if has_term_frequencies else "NULL"
        rows = connection.execute(
            f"""
            SELECT collection, vector_id, content, metadata, {term_column}
            FROM vectors
            """
        ).fetchall()
        connection.execute("DROP TABLE IF EXISTS vectors_rebuilt")
        self._create_table(connection, "vectors_rebuilt")
        migrated_rows = [
            (
                collection,
                vector_id,
                content,
                metadata,
                term_frequencies or json.dumps(self._term_frequencies(content)),
            )
            for collection, vector_id, content, metadata, term_frequencies in rows
        ]
        connection.executemany(
            """
            INSERT INTO vectors_rebuilt
                (collection, vector_id, content, metadata, term_frequencies)
            VALUES (?, ?, ?, ?, ?)
            """,
            migrated_rows,
        )
        connection.execute("DROP TABLE vectors")
        connection.execute("ALTER TABLE vectors_rebuilt RENAME TO vectors")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def replace_documents(self, documents: Sequence[Document]) -> int:
        """Build sparse term vectors, then atomically replace this collection."""
        if not documents:
            raise ValueError("Cannot replace the collection with no documents.")

        rows = []
        for document in documents:
            term_frequencies = self._term_frequencies(document.page_content)
            rows.append(
                (
                    self.collection,
                    document.id or str(uuid4()),
                    document.page_content,
                    json.dumps(document.metadata),
                    json.dumps(term_frequencies),
                )
            )

        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM vectors WHERE collection = ?", (self.collection,)
            )
            connection.executemany(
                """
                INSERT INTO vectors
                    (collection, vector_id, content, metadata, term_frequencies)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """Return the highest cosine-similarity documents from this collection."""
        if k < 1:
            raise ValueError("k must be at least 1.")
        query_terms = self._term_frequencies(query)
        if not query_terms:
            return []

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT content, metadata, term_frequencies
                FROM vectors
                WHERE collection = ?
                """,
                (self.collection,),
            ).fetchall()

        term_vectors = [
            json.loads(term_frequencies)
            if term_frequencies
            else self._term_frequencies(content)
            for content, _, term_frequencies in rows
        ]
        document_count = len(term_vectors)
        document_frequencies: dict[str, int] = {}
        for term_vector in term_vectors:
            for term in term_vector:
                document_frequencies[term] = document_frequencies.get(term, 0) + 1

        query_vector = {
            term: frequency
            * (math.log((document_count + 1) / (document_frequencies.get(term, 0) + 1)) + 1)
            for term, frequency in query_terms.items()
        }
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        if query_norm == 0:
            return []

        matches: list[tuple[Document, float]] = []
        for (content, metadata_json, _), term_frequencies in zip(
            rows, term_vectors, strict=True
        ):
            vector = {
                term: frequency
                * (math.log((document_count + 1) / (document_frequencies[term] + 1)) + 1)
                for term, frequency in term_frequencies.items()
            }
            vector_norm = math.sqrt(sum(value * value for value in vector.values()))
            if vector_norm == 0:
                continue
            cosine_similarity = sum(
                value * vector.get(term, 0.0) for term, value in query_vector.items()
            ) / (query_norm * vector_norm)
            if cosine_similarity <= 0:
                continue
            document = Document(
                page_content=content,
                metadata=json.loads(metadata_json),
            )
            matches.append(
                (document, max(0.0, min(1.0, cosine_similarity)))
            )

        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:k]

    @staticmethod
    def _term_frequencies(text: str) -> dict[str, int]:
        return dict(
            Counter(
                _normalize_term(token)
                for token in re.findall(r"[a-z0-9]+", text.lower())
                if token not in _STOP_WORDS
            )
        )

    def count(self) -> int:
        """Return the number of vectors stored in this collection."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM vectors WHERE collection = ?",
                (self.collection,),
            ).fetchone()
        return int(row[0])


def _normalize_term(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    if len(term) > 4 and term.endswith("s"):
        return term[:-1]
    return term
