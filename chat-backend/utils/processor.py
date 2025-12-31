from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Dict, List
import chromadb

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))

from data_preprocessing import process_document
from rag_pipeline import get_embedding_function

class DocumentProcessor:
    def __init__(self, chroma_persist_dir: Path):
        self.chroma_persist_dir = chroma_persist_dir
        self.client = chromadb.PersistentClient(path=str(chroma_persist_dir))
    
    def process_session_files(
        self,
        session_dir: Path,
        collection_name: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> Dict:
        """
        Process all files in session directory and ingest into ChromaDB
        """
        try:
            # Find PDF files
            pdf_files = list(session_dir.glob('*.pdf'))
            if not pdf_files:
                return {
                    'status': 'error',
                    'error': 'No PDF files found in upload'
                }
            
            # Process each document
            processed_docs = []
            all_chunks = []
            
            for pdf_path in pdf_files:
                json_path = session_dir / f"{pdf_path.stem}.json"
                
                # Use existing preprocessing function
                doc = process_document(
                    pdf_path=pdf_path,
                    json_path=json_path,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
                
                processed_docs.append({
                    'doc_id': doc['doc_id'],
                    'num_chunks': len(doc['chunks'])
                })
                
                all_chunks.extend(doc['chunks'])
            
            # Ingest into ChromaDB
            emb_fn = get_embedding_function('sentence-transformers', 'all-MiniLM-L6-v2')
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=emb_fn
            )
            
            # Prepare data for ingestion
            ids = [chunk['id'] for chunk in all_chunks]
            documents = [chunk['text'] for chunk in all_chunks]
            metadatas = [chunk['metadata'] for chunk in all_chunks]
            
            # Ingest in batches
            batch_size = 500
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas
                )
            
            return {
                'status': 'success',
                'collection_name': collection_name,
                'num_documents': len(processed_docs),
                'num_chunks': len(all_chunks),
                'documents': processed_docs
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def list_collections(self) -> List[str]:
        """List all ChromaDB collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a ChromaDB collection"""
        try:
            self.client.delete_collection(name=collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def get_collection(self, collection_name: str):
        """Get a ChromaDB collection"""
        try:
            emb_fn = get_embedding_function('sentence-transformers', 'all-MiniLM-L6-v2')
            return self.client.get_collection(
                name=collection_name,
                embedding_function=emb_fn
            )
        except Exception as e:
            print(f"Error getting collection: {e}")
            return None
