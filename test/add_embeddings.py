#!/usr/bin/env python3
"""Create or replace MongoDB embeddings using the configured model.

Use ``--overwrite`` when changing embedding models. Mixing vectors from
different models in one Atlas vector index makes retrieval invalid even when
their dimensions are identical.
"""

import argparse
import os
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

load_dotenv()


def batched(items: Iterable[dict], size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create MongoDB embeddings with the configured embedding model"
    )
    parser.add_argument(
        "--collection", default=os.getenv("MONGO_COLLECTION_NAME", "mental")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing embedding fields"
    )
    args = parser.parse_args()

    uri = os.getenv("MONGO_URI")
    if not uri:
        raise SystemExit("MONGO_URI is required")
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    use_local = os.getenv("USE_LOCAL_EMBEDDING", "true").lower() in {"1", "true", "yes"}
    mongo = MongoClient(uri)
    collection = mongo[os.getenv("MONGO_DB_NAME", "Data")][args.collection]

    if use_local:
        if SentenceTransformer is None:
            raise SystemExit(
                "sentence-transformers is required when USE_LOCAL_EMBEDDING=true"
            )
        print(f"Loading local embedding model: {model_name}")
        encoder = SentenceTransformer(model_name, trust_remote_code=True)

        def embed(texts: list[str]) -> list[list[float]]:
            return encoder.encode(texts, normalize_embeddings=True).tolist()
    else:
        endpoint = os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:1234/v1")
        client = OpenAI(base_url=endpoint, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
        print(f"Using embedding endpoint: {endpoint}; model: {model_name}")

        def embed(texts: list[str]) -> list[list[float]]:
            return [
                item.embedding
                for item in client.embeddings.create(model=model_name, input=texts).data
            ]

    query = {} if args.overwrite else {"embedding": {"$exists": False}}
    total = collection.count_documents(query)
    print(
        f"Collection={args.collection}; candidates={total}; overwrite={args.overwrite}"
    )
    updated = 0
    try:
        for docs in tqdm(
            batched(
                collection.find(query, {"content": 1, "summary": 1}).batch_size(
                    args.batch_size
                ),
                args.batch_size,
            ),
            total=(total + args.batch_size - 1) // args.batch_size,
        ):
            usable = [
                (doc, doc.get("content") or doc.get("summary") or "") for doc in docs
            ]
            usable = [(doc, text) for doc, text in usable if text.strip()]
            if not usable:
                continue
            vectors = embed([text for _, text in usable])
            if any(len(vector) != 1024 for vector in vectors):
                raise RuntimeError(
                    "Expected 1024-dimensional BGE-M3 vectors; check EMBEDDING_MODEL and Atlas index configuration"
                )
            operations = [
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": vector, "embedding_model": model_name}},
                )
                for (doc, _), vector in zip(usable, vectors)
            ]
            collection.bulk_write(operations, ordered=False)
            updated += len(operations)
    finally:
        mongo.close()
    print(f"Updated {updated} embedding vectors using {model_name}.")


if __name__ == "__main__":
    main()
