#!/usr/bin/env python3
"""Backfill explicit source_tier on existing Mongo chunks.

Defaults to dry-run. The caller must select the tier after reviewing the
source set; this tool never infers clinical authority from a chunk's text.
"""
import argparse
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
VALID_SOURCE_TIERS = ("tier_1", "tier_2", "tier_3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill source_tier for unlabelled MongoDB chunks")
    parser.add_argument("--source-tier", required=True, choices=VALID_SOURCE_TIERS)
    parser.add_argument("--collection", default=os.getenv("MONGO_COLLECTION_NAME", "mental"))
    parser.add_argument("--apply", action="store_true", help="Actually update documents; omit for dry-run")
    args = parser.parse_args()
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise SystemExit("MONGO_URI is required")
    client = MongoClient(uri)
    try:
        collection = client[os.getenv("MONGO_DB_NAME", "Data")][args.collection]
        query = {"$and": [
            {"$or": [{"source_tier": {"$exists": False}}, {"source_tier": None}, {"source_tier": ""}]},
            {"$or": [{"tier": {"$exists": False}}, {"tier": None}, {"tier": ""}]},
        ]}
        count = collection.count_documents(query)
        print(f"{count} unlabelled documents in {args.collection} would receive {args.source_tier}.")
        if args.apply and count:
            result = collection.update_many(query, {"$set": {"source_tier": args.source_tier}})
            print(f"Updated {result.modified_count} documents.")
        elif not args.apply:
            print("Dry run only. Re-run with --apply after confirming the source tier.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
