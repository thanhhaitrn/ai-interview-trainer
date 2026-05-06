import argparse
import sqlite3
from pathlib import Path

from qdrant_client import QdrantClient

from src.db.embeddings import MODEL_NAME, embed_documents
from src.db.qdrant_store import (
    delete_chunk_table_points,
    ensure_collection,
    ensure_payload_indexes,
    get_collection_name,
    get_qdrant_client,
    make_point,
)

DB_PATH = Path("db/resume_kb.db")
CHUNK_TABLES = {"resume_chunks", "job_chunks"}


def fetch_chunks(conn: sqlite3.Connection, chunk_table: str) -> list[sqlite3.Row]:
    if chunk_table not in CHUNK_TABLES:
        raise ValueError(f"Unsupported chunk table: {chunk_table}")

    owner_id_column = "resume_id" if chunk_table == "resume_chunks" else "job_id"

    return conn.execute(
        f"""
        SELECT
            id,
            {owner_id_column},
            section_type,
            source_table,
            source_id,
            chunk_text
        FROM {chunk_table} c
        ORDER BY c.id
        """
    ).fetchall()


def embed_table(
    conn: sqlite3.Connection,
    client: QdrantClient,
    chunk_table: str,
    *,
    batch_size: int,
    collection_name: str,
    recreate_collection: bool,
    collection_ready: bool,
) -> tuple[int, bool]:
    rows = fetch_chunks(conn, chunk_table)
    total = 0

    if not rows and client.collection_exists(collection_name):
        ensure_payload_indexes(client, collection_name)
        delete_chunk_table_points(
            client,
            collection_name,
            chunk_table=chunk_table,
            model_name=MODEL_NAME,
        )
        return total, collection_ready

    for start in range(0, len(rows), batch_size):
        batch = rows[start: start + batch_size]
        texts = [row["chunk_text"] for row in batch]
        vectors = embed_documents(texts)

        if not collection_ready:
            ensure_collection(
                client,
                collection_name,
                len(vectors[0]),
                recreate=recreate_collection,
            )
            collection_ready = True

        if start == 0:
            delete_chunk_table_points(
                client,
                collection_name,
                chunk_table=chunk_table,
                model_name=MODEL_NAME,
            )

        client.upsert(
            collection_name=collection_name,
            points=[
                make_point(
                    row,
                    vector,
                    chunk_table=chunk_table,
                    model_name=MODEL_NAME,
                )
                for row, vector in zip(batch, vectors)
            ],
            wait=True,
        )

        total += len(batch)

    return total, collection_ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--collection",
        default=get_collection_name(),
        help="Qdrant collection name. Default: QDRANT_COLLECTION or resume_kb_chunks.",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Delete and recreate the Qdrant collection before uploading vectors.",
    )
    args = parser.parse_args()

    with sqlite3.connect(args.db) as conn:
        conn.row_factory = sqlite3.Row
        client = get_qdrant_client()

        resume_count, collection_ready = embed_table(
            conn,
            client,
            "resume_chunks",
            batch_size=args.batch_size,
            collection_name=args.collection,
            recreate_collection=args.recreate_collection,
            collection_ready=False,
        )
        job_count, _ = embed_table(
            conn,
            client,
            "job_chunks",
            batch_size=args.batch_size,
            collection_name=args.collection,
            recreate_collection=args.recreate_collection and not collection_ready,
            collection_ready=collection_ready,
        )

    print(f"Embedded {resume_count} resume chunks.")
    print(f"Embedded {job_count} job chunks.")
    print(f"qdrant_collection={args.collection}")


if __name__ == "__main__":
    main()
