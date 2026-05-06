import argparse

from qdrant_client import models

from src.db.embeddings import MODEL_NAME, embed_query
from src.db.qdrant_store import (
    ensure_payload_indexes,
    get_collection_name,
    get_qdrant_client,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--collection",
        default=get_collection_name(),
        help="Qdrant collection name. Default: QDRANT_COLLECTION or resume_kb_chunks.",
    )
    args = parser.parse_args()

    query_vector = embed_query(args.query)
    client = get_qdrant_client()
    ensure_payload_indexes(client, args.collection)

    response = client.query_points(
        collection_name=args.collection,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_table",
                    match=models.MatchValue(value="resume_chunks"),
                ),
                models.FieldCondition(
                    key="model",
                    match=models.MatchValue(value=MODEL_NAME),
                ),
            ]
        ),
        limit=args.top_k,
        with_payload=True,
    )

    points = getattr(response, "points", response)

    for point in points:
        payload = point.payload or {}
        print("=" * 80)
        print(f"score={point.score:.4f}")
        print(f"resume_id={payload.get('resume_id')}")
        print(f"section_type={payload.get('section_type')}")
        print(payload.get("chunk_text", ""))


if __name__ == "__main__":
    main()
