"""MongoDB adapter for selecting unused source chunks."""
import os
from typing import Any, Dict, List, Optional
from pymongo import MongoClient


def select_unused_anchor(used_anchor_ids: List[str]) -> Optional[Dict[str, Any]]:
    uri = os.getenv("MONGO_URI")
    if not uri:
        return None
    mongo = MongoClient(uri)
    try:
        result = list(mongo[os.getenv("MONGO_DB_NAME", "Data")][os.getenv("MONGO_COLLECTION_NAME", "mental")].aggregate([
            {"$match": {"uuid": {"$nin": used_anchor_ids}}}, {"$sample": {"size": 1}},
        ]))
    finally:
        mongo.close()
    if not result:
        return None
    doc = result[0]
    return {"chunk_id": str(doc.get("chunk_id") or doc.get("uuid") or doc.get("_id")),
            "title": doc.get("title") or " / ".join(doc.get("headers", [])),
            "summary": doc.get("summary", ""), "content": doc.get("content", ""),
            "keywords": doc.get("keywords", []), "type": doc.get("type", ""),
            "tier": doc.get("tier") or doc.get("source_tier")}
