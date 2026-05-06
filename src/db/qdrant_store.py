import os
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models


DEFAULT_COLLECTION_NAME = "resume_kb_chunks"
ENV_PATH = Path(".env")
FILTER_PAYLOAD_INDEXES = {
    "chunk_table": models.PayloadSchemaType.KEYWORD,
    "model": models.PayloadSchemaType.KEYWORD,
}


def load_env_file(path: Path = ENV_PATH) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding the shell env."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def get_collection_name() -> str:
    load_env_file()
    return os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)


def get_qdrant_client() -> QdrantClient:
    load_env_file()

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise RuntimeError("Missing QDRANT_URL in environment or .env.")

    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    *,
    recreate: bool = False,
) -> None:
    vectors_config = models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
    )

    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )

    ensure_payload_indexes(client, collection_name)


def ensure_payload_indexes(
    client: QdrantClient,
    collection_name: str,
) -> None:
    collection_info = client.get_collection(collection_name)
    payload_schema = collection_info.payload_schema or {}

    for field_name, field_schema in FILTER_PAYLOAD_INDEXES.items():
        if field_name in payload_schema:
            continue

        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )


def make_point_id(chunk_table: str, chunk_id: int, model_name: str) -> str:
    point_key = f"{model_name}:{chunk_table}:{chunk_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, point_key))


def make_chunk_payload(
    row: Any,
    *,
    chunk_table: str,
    model_name: str,
) -> dict[str, Any]:
    payload = {
        "chunk_table": chunk_table,
        "chunk_id": int(row["id"]),
        "section_type": row["section_type"],
        "source_table": row["source_table"],
        "source_id": row["source_id"],
        "chunk_text": row["chunk_text"],
        "model": model_name,
    }

    if "resume_id" in row.keys():
        payload["resume_id"] = int(row["resume_id"])

    if "job_id" in row.keys():
        payload["job_id"] = int(row["job_id"])

    return {
        key: value
        for key, value in payload.items()
        if value is not None
    }


def make_point(
    row: Any,
    vector: list[float],
    *,
    chunk_table: str,
    model_name: str,
) -> models.PointStruct:
    return models.PointStruct(
        id=make_point_id(chunk_table, int(row["id"]), model_name),
        vector=vector,
        payload=make_chunk_payload(
            row,
            chunk_table=chunk_table,
            model_name=model_name,
        ),
    )


def delete_chunk_table_points(
    client: QdrantClient,
    collection_name: str,
    *,
    chunk_table: str,
    model_name: str,
) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_table",
                        match=models.MatchValue(value=chunk_table),
                    ),
                    models.FieldCondition(
                        key="model",
                        match=models.MatchValue(value=model_name),
                    ),
                ]
            )
        ),
    )
