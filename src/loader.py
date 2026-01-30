
import json
from pathlib import Path
from typing import List, Dict, Any
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

class Loader:
    """
    Loads data from MongoDB instead of local files.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize MongoDB connection.
        data_dir parameter kept for backward compatibility but not used.
        """
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_NAME", "Data")
        self.collection_name = os.getenv("MONGO_COLLECTION_NAME", "mental")
        
        if not self.mongo_uri:
            raise ValueError("MONGO_URI not set in environment")
        
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        
    def load_files(self) -> List[Dict[str, Any]]:
        """
        Load all documents from MongoDB collection.
        
        Returns:
            List of document dictionaries
        """
        all_data = list(self.collection.find({}, {"_id": 0}))  # Exclude MongoDB _id
        print(f"Loaded {len(all_data)} entries from MongoDB collection '{self.collection_name}'")
        return all_data
