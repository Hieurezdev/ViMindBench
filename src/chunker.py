
from typing import List, Dict, Any

class Chunker:
    """
    Handles text chunking. 
    Since the input JSON already contains 'content' which appears to be chunked/paragraph-sized,
    this class primarily verifies or standardizes the chunks.
    """
    
    def __init__(self):
        pass
        
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes the loaded data and ensures the 'content' field is ready for retrieval.
        Returns a list of dicts with at least 'content' and 'metadata'.
        """
        processed_chunks = []
        for item in data:
            content = item.get('content', '')
            if not content:
                continue
                
            # Here we could split larger contents if necessary.
            # For now, we assume the input granularity is sufficient.
            
            chunk = {
                'page_content': content,
                'metadata': {
                    'uuid': item.get('uuid'),
                    'headers': item.get('headers'),
                    'summary': item.get('summary'),
                    'keywords': item.get('keywords'),
                    'type': item.get('type')
                }
            }
            processed_chunks.append(chunk)
            
        return processed_chunks
