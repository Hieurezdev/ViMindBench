#!/usr/bin/env python3
"""
Script to automatically import all JSON files into MongoDB
Author: Auto-generated
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
import argparse
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, ConnectionFailure
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
VALID_SOURCE_TIERS = ("tier_1", "tier_2", "tier_3")

# Load environment variables
load_dotenv()


class JSONToMongoImporter:
    """Import JSON files to MongoDB"""
    
    def __init__(self, mongo_uri: str = None, database_name: str = None):
        """
        Initialize MongoDB connection
        
        Args:
            mongo_uri: MongoDB connection URI (default from env MONGO_URI)
            database_name: Database name (default from env MONGO_DB_NAME or 'finetuning_data')
        """
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.database_name = database_name or os.getenv("MONGO_DB_NAME", "finetuning_data")
        
        try:
            self.client = MongoClient(self.mongo_uri)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            logger.info(f"✓ Connected to MongoDB at {self.mongo_uri}")
            logger.info(f"✓ Using database: {self.database_name}")
        except ConnectionFailure as e:
            logger.error(f"✗ Failed to connect to MongoDB: {e}")
            raise
    
    def find_json_files(self, directory: str) -> List[Path]:
        """
        Find all JSON files in directory and subdirectories
        
        Args:
            directory: Path to directory to search
            
        Returns:
            List of Path objects for JSON files
        """
        json_files = []
        search_path = Path(directory)
        
        if not search_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return json_files
        
        # Find all .json files recursively
        json_files = list(search_path.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {directory}")
        
        return json_files
    
    def load_json_file(self, file_path: Path, source_tier: str | None = None) -> List[Dict[str, Any]]:
        """
        Load JSON file and return data as list
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of documents to insert
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # If data is a dict, wrap it in a list
            if isinstance(data, dict):
                data = [data]
            # If data is already a list, use as is
            elif isinstance(data, list):
                pass
            else:
                logger.warning(f"Unexpected data type in {file_path}: {type(data)}")
                return []
            
            # Add source file metadata to each document
            for item in data:
                if isinstance(item, dict):
                    item['_source_file'] = str(file_path.name)
                    item['_source_path'] = str(file_path)
                    # Source tier is explicit provenance, never inferred from text.
                    if source_tier and not item.get('source_tier'):
                        item['source_tier'] = source_tier
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"✗ Failed to parse JSON file {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"✗ Error loading file {file_path}: {e}")
            return []
    
    def import_to_collection(self, documents: List[Dict[str, Any]], 
                            collection_name: str) -> int:
        """
        Import documents to MongoDB collection
        
        Args:
            documents: List of documents to insert
            collection_name: Name of collection
            
        Returns:
            Number of documents inserted
        """
        if not documents:
            return 0
        
        collection = self.db[collection_name]
        
        try:
            result = collection.insert_many(documents, ordered=False)
            inserted_count = len(result.inserted_ids)
            logger.info(f"✓ Inserted {inserted_count} documents into '{collection_name}'")
            return inserted_count
            
        except BulkWriteError as e:
            # Some documents may have been inserted
            inserted_count = e.details.get('nInserted', 0)
            logger.warning(f"Partial insert: {inserted_count} documents inserted, "
                         f"{len(e.details.get('writeErrors', []))} errors")
            return inserted_count
            
        except Exception as e:
            logger.error(f"✗ Failed to insert documents: {e}")
            return 0
    
    def import_directory(self, directory: str, collection_name: str = None,
                        batch_size: int = 1000, source_tier: str | None = None) -> Dict[str, Any]:
        """
        Import all JSON files from directory to MongoDB
        
        Args:
            directory: Directory containing JSON files
            collection_name: Collection name (default: 'json_data')
            batch_size: Number of documents to insert in each batch
            
        Returns:
            Dictionary with import statistics
        """
        collection_name = collection_name or os.getenv("MONGO_COLLECTION_NAME", "json_data")
        
        logger.info(f"Starting import from directory: {directory}")
        logger.info(f"Target collection: {collection_name}")
        
        json_files = self.find_json_files(directory)
        
        if not json_files:
            logger.warning("No JSON files found!")
            return {"files_processed": 0, "total_documents": 0}
        
        total_documents = 0
        files_processed = 0
        batch = []
        
        for file_path in json_files:
            logger.info(f"Processing: {file_path.name}")
            
            documents = self.load_json_file(file_path, source_tier=source_tier)
            
            if documents:
                batch.extend(documents)
                files_processed += 1
                
                # Insert in batches
                if len(batch) >= batch_size:
                    inserted = self.import_to_collection(batch, collection_name)
                    total_documents += inserted
                    batch = []
        
        # Insert remaining documents
        if batch:
            inserted = self.import_to_collection(batch, collection_name)
            total_documents += inserted
        
        stats = {
            "files_processed": files_processed,
            "total_files": len(json_files),
            "total_documents": total_documents
        }
        
        logger.info("=" * 60)
        logger.info("Import Summary:")
        logger.info(f"  Files processed: {stats['files_processed']}/{stats['total_files']}")
        logger.info(f"  Documents inserted: {stats['total_documents']}")
        logger.info("=" * 60)
        
        return stats
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Import JSON chunks into MongoDB with source provenance")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "data/formated_data"))
    parser.add_argument("--collection", default=os.getenv("MONGO_COLLECTION_NAME", "Data"))
    parser.add_argument("--source-tier", choices=VALID_SOURCE_TIERS, default=os.getenv("SOURCE_TIER"),
                        help="Tier assigned to this imported source set; do not guess this from chunk text")
    args = parser.parse_args()
    # Configuration
    DATA_DIR = args.data_dir
    COLLECTION_NAME = args.collection
    
    # Initialize importer
    importer = JSONToMongoImporter()
    
    try:
        # Import all JSON files
        stats = importer.import_directory(
            directory=DATA_DIR,
            collection_name=COLLECTION_NAME,
            batch_size=1000,
            source_tier=args.source_tier,
        )
        
        logger.info("\n✓ Import completed successfully!")
        
    except Exception as e:
        logger.error(f"\n✗ Import failed: {e}")
        raise
    finally:
        importer.close()


if __name__ == "__main__":
    main()
