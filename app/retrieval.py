"""Context retrieval helpers for automatic question-generation enrichment."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path("db/resume_kb.db")
DEFAULT_TOP_K = 6


def retrieval_enabled() -> bool:
    """Return True when automatic table retrieval is enabled."""
    value = os.getenv("INTERVIEW_AUTO_RETRIEVAL", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def retrieval_top_k() -> int:
    """Read retrieval depth from env while keeping a safe range."""
    raw_value = os.getenv("INTERVIEW_RETRIEVAL_TOP_K", str(DEFAULT_TOP_K))

    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_TOP_K

    return max(1, min(parsed, 20))


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """Keep unique non-empty lines while preserving insertion order."""
    seen: set[str] = set()
    unique_items: list[str] = []

    for item in items:
        line = " ".join(str(item).split()).strip()
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        unique_items.append(line)

    return unique_items


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _build_query_text(
    job_description_context: list[str],
    interview_type: str,
    difficulty: str,
) -> str:
    parts = [
        "Interview question retrieval query",
        f"Interview type: {interview_type}",
        f"Difficulty: {difficulty}",
        "Job context:",
        "\n".join(job_description_context),
    ]
    return "\n".join(part for part in parts if part.strip())


def _extract_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]{3,}", text.lower())
    }


def _chunk_overlap_score(chunk_text: str, query_terms: set[str]) -> int:
    if not query_terms:
        return 0

    chunk_terms = _extract_terms(chunk_text)
    return len(chunk_terms & query_terms)


def _resolve_resume_id(
    conn: sqlite3.Connection,
    resume_file_name: str | None,
) -> int | None:
    if not _table_exists(conn, "resumes_doc"):
        return None

    if resume_file_name:
        row = conn.execute(
            """
            SELECT id
            FROM resumes_doc
            WHERE file_name = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (resume_file_name,),
        ).fetchone()
        if row is not None:
            return int(row["id"])

    row = conn.execute(
        """
        SELECT id
        FROM resumes_doc
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None

    return int(row["id"])


def _resolve_job_id(
    conn: sqlite3.Connection,
    role_title: str | None,
    company: str | None,
) -> int | None:
    if not _table_exists(conn, "job_descriptions"):
        return None

    if role_title and company:
        row = conn.execute(
            """
            SELECT id
            FROM job_descriptions
            WHERE job_title = ?
              AND company = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (role_title, company),
        ).fetchone()
        if row is not None:
            return int(row["id"])

    if role_title:
        row = conn.execute(
            """
            SELECT id
            FROM job_descriptions
            WHERE job_title = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (role_title,),
        ).fetchone()
        if row is not None:
            return int(row["id"])

    row = conn.execute(
        """
        SELECT id
        FROM job_descriptions
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None

    return int(row["id"])


def _select_sqlite_chunks(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    owner_id_column: str,
    owner_id: int | None,
    query_text: str,
    top_k: int,
) -> list[str]:
    if not _table_exists(conn, table_name):
        return []

    if owner_id is not None:
        rows = conn.execute(
            f"""
            SELECT id, chunk_text
            FROM {table_name}
            WHERE {owner_id_column} = ?
            ORDER BY id DESC
            """,
            (owner_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            SELECT id, chunk_text
            FROM {table_name}
            ORDER BY id DESC
            """
        ).fetchall()

    if not rows:
        return []

    query_terms = _extract_terms(query_text)
    scored = []
    for row in rows:
        chunk_text = str(row["chunk_text"]).strip()
        score = _chunk_overlap_score(chunk_text, query_terms)
        scored.append((score, int(row["id"]), chunk_text))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [chunk_text for _, _, chunk_text in scored[:top_k]]
    return _dedupe_keep_order(selected)


def retrieve_from_sqlite(
    *,
    db_path: Path,
    query_text: str,
    resume_file_name: str | None,
    role_title: str | None,
    company: str | None,
    top_k: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return retrieved resume/job chunks from SQLite with lexical ranking."""
    if not db_path.exists():
        return [], [], {"source": "sqlite", "used": False, "reason": "db_missing"}

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        resume_id = _resolve_resume_id(conn, resume_file_name)
        job_id = _resolve_job_id(conn, role_title, company)

        resume_chunks = _select_sqlite_chunks(
            conn,
            table_name="resume_chunks",
            owner_id_column="resume_id",
            owner_id=resume_id,
            query_text=query_text,
            top_k=top_k,
        )
        job_chunks = _select_sqlite_chunks(
            conn,
            table_name="job_chunks",
            owner_id_column="job_id",
            owner_id=job_id,
            query_text=query_text,
            top_k=top_k,
        )

    metadata = {
        "source": "sqlite",
        "used": bool(resume_chunks or job_chunks),
        "resume_id": resume_id,
        "job_id": job_id,
    }
    return resume_chunks, job_chunks, metadata


def _query_qdrant_chunk_table(
    *,
    client: Any,
    collection_name: str,
    query_vector: list[float],
    chunk_table: str,
    model_name: str,
    owner_key: str,
    owner_id: int | None,
    top_k: int,
    models_module: Any,
) -> list[str]:
    conditions = [
        models_module.FieldCondition(
            key="chunk_table",
            match=models_module.MatchValue(value=chunk_table),
        ),
        models_module.FieldCondition(
            key="model",
            match=models_module.MatchValue(value=model_name),
        ),
    ]

    if owner_id is not None:
        conditions.append(
            models_module.FieldCondition(
                key=owner_key,
                match=models_module.MatchValue(value=owner_id),
            )
        )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=models_module.Filter(must=conditions),
        limit=top_k,
        with_payload=True,
    )

    points = getattr(response, "points", response)
    lines: list[str] = []

    for point in points:
        payload = getattr(point, "payload", None) or {}
        chunk_text = str(payload.get("chunk_text", "")).strip()
        if chunk_text:
            lines.append(chunk_text)

    return _dedupe_keep_order(lines)


def retrieve_from_qdrant(
    *,
    query_text: str,
    resume_id: int | None,
    job_id: int | None,
    top_k: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return retrieved resume/job chunks from Qdrant when configured."""
    try:
        from qdrant_client import models as qdrant_models
        from src.db.embeddings import MODEL_NAME, embed_query
        from src.db.qdrant_store import (
            ensure_payload_indexes,
            get_collection_name,
            get_qdrant_client,
        )
    except Exception as error:
        return [], [], {
            "source": "qdrant",
            "used": False,
            "reason": f"import_error:{error.__class__.__name__}",
        }

    try:
        query_vector = embed_query(query_text)
        client = get_qdrant_client()
        collection_name = get_collection_name()

        if not client.collection_exists(collection_name):
            return [], [], {
                "source": "qdrant",
                "used": False,
                "reason": "collection_missing",
            }

        ensure_payload_indexes(client, collection_name)

        resume_chunks = _query_qdrant_chunk_table(
            client=client,
            collection_name=collection_name,
            query_vector=query_vector,
            chunk_table="resume_chunks",
            model_name=MODEL_NAME,
            owner_key="resume_id",
            owner_id=resume_id,
            top_k=top_k,
            models_module=qdrant_models,
        )
        job_chunks = _query_qdrant_chunk_table(
            client=client,
            collection_name=collection_name,
            query_vector=query_vector,
            chunk_table="job_chunks",
            model_name=MODEL_NAME,
            owner_key="job_id",
            owner_id=job_id,
            top_k=top_k,
            models_module=qdrant_models,
        )

        return resume_chunks, job_chunks, {
            "source": "qdrant",
            "used": bool(resume_chunks or job_chunks),
            "collection": collection_name,
        }
    except Exception as error:
        return [], [], {
            "source": "qdrant",
            "used": False,
            "reason": f"query_error:{error.__class__.__name__}",
        }


def _merge_context_lines(
    base_lines: list[str],
    retrieved_lines: list[str],
    *,
    prefix: str,
) -> list[str]:
    merged = list(base_lines)
    for line in retrieved_lines:
        merged.append(f"{prefix}: {line}")
    return _dedupe_keep_order(merged)


def retrieve_table_context_for_question(
    *,
    cv_context: list[str],
    job_description_context: list[str],
    interview_type: str,
    difficulty: str,
    resume_file_name: str | None = None,
    job_role_title: str | None = None,
    job_company: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Enrich question-generation context with retrieved chunks by default.

    Qdrant semantic retrieval is attempted first. If unavailable or empty, SQLite
    lexical retrieval is used as a fallback.
    """
    if not retrieval_enabled():
        return {
            "cv_context": list(cv_context),
            "job_description_context": list(job_description_context),
            "metadata": {"enabled": False},
        }

    resolved_db_path = Path(
        db_path or os.getenv("INTERVIEW_DB_PATH", str(DEFAULT_DB_PATH))
    )
    top_k = retrieval_top_k()
    query_text = _build_query_text(job_description_context, interview_type, difficulty)

    sqlite_resume_chunks, sqlite_job_chunks, sqlite_meta = retrieve_from_sqlite(
        db_path=resolved_db_path,
        query_text=query_text,
        resume_file_name=resume_file_name,
        role_title=job_role_title,
        company=job_company,
        top_k=top_k,
    )

    qdrant_resume_chunks, qdrant_job_chunks, qdrant_meta = retrieve_from_qdrant(
        query_text=query_text,
        resume_id=sqlite_meta.get("resume_id"),
        job_id=sqlite_meta.get("job_id"),
        top_k=top_k,
    )

    selected_resume_chunks = qdrant_resume_chunks or sqlite_resume_chunks
    selected_job_chunks = qdrant_job_chunks or sqlite_job_chunks

    enriched_cv_context = _merge_context_lines(
        cv_context,
        selected_resume_chunks,
        prefix="Retrieved resume evidence",
    )
    enriched_job_context = _merge_context_lines(
        job_description_context,
        selected_job_chunks,
        prefix="Retrieved job evidence",
    )

    return {
        "cv_context": enriched_cv_context,
        "job_description_context": enriched_job_context,
        "metadata": {
            "enabled": True,
            "db_path": str(resolved_db_path),
            "top_k": top_k,
            "qdrant": qdrant_meta,
            "sqlite": sqlite_meta,
            "resume_chunks_added": len(selected_resume_chunks),
            "job_chunks_added": len(selected_job_chunks),
        },
    }

