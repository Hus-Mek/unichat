"""
RAG Engine Module
Core RAG functionality - embeddings, vector DB, retrieval
"""

from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from .config import Config
from .document_processor import DocumentProcessor
from .access_control import AccessController


class RAGEngine:
    """Main RAG engine handling embeddings and retrieval"""
    
    def __init__(self):
        """Initialize RAG engine"""
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.access_controller = AccessController()
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(Config.EMBED_MODEL)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            Config.COLLECTION_NAME
        )
    
    def index_document(
        self,
        file_bytes: bytes,
        file_name: str,
        file_extension: str,
        access_level: str,
        owner: Optional[str] = None
    ) -> Dict:
        """
        Index a document with access control metadata
        
        Args:
            file_bytes: Document file as bytes
            file_name: Original file name
            file_extension: File extension (.pdf, .xlsx)
            access_level: Access level (public, student, faculty)
            owner: Optional owner ID for personal documents
            
        Returns:
            Dictionary with indexing results
        """
        try:
            # Extract text
            text = self.doc_processor.process_document(
                file_bytes, 
                file_extension
            )
            
            if not text:
                return {
                    "success": False,
                    "error": "No text extracted from document"
                }
            
            # Chunk text
            chunks = self.doc_processor.chunk_text(text)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "Failed to create chunks"
                }
            
            # Generate embeddings
            embeddings = self.embedder.encode(
                chunks, 
                show_progress_bar=True
            ).tolist()
            
            # Prepare for ChromaDB
            chunk_ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_name}_{i}"
                chunk_ids.append(chunk_id)
                
                metadata = {
                    "access_level": access_level,
                    "source": file_name,
                    "chunk_index": i
                }
                
                # Add owner if provided
                if owner:
                    metadata["owner"] = owner
                else:
                    metadata["owner"] = ""  # ChromaDB doesn't like None
                
                metadatas.append(metadata)
            
            # Add to ChromaDB
            self.collection.add(
                ids=chunk_ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return {
                "success": True,
                "file_name": file_name,
                "chunks": len(chunks),
                "access_level": access_level
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def query(
        self,
        question: str,
        user_level: str,
        user_id: Optional[str] = None,
        n_results: int = None
    ) -> Dict:
        """
        Query the RAG system with access control
        
        Args:
            question: User's question
            user_level: User's access level
            user_id: Optional user ID for personal documents
            n_results: Number of results to retrieve
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        if n_results is None:
            n_results = Config.DEFAULT_N_RESULTS
        
        try:
            # Build access filter
            where_filter = self.access_controller.build_filter(
                user_level, 
                user_id
            )
            
            # Generate query embedding
            query_embedding = self.embedder.encode([question])[0].tolist()
            
            # Query ChromaDB with filter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter  # ← Access control!
            )
            
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            
            if not documents:
                return {
                    "success": True,
                    "documents": [],
                    "context": "",
                    "sources": [],
                    "message": "No accessible documents found"
                }
            
            # Build context
            context = "\n\n".join(documents)
            
            # Extract sources with detailed breakdown
            sources_detail = {}
            for meta in metadatas:
                source = meta.get("source", "unknown")
                access_level = meta.get("access_level", "unknown")
                if source not in sources_detail:
                    sources_detail[source] = {
                        "count": 0,
                        "access_level": access_level
                    }
                sources_detail[source]["count"] += 1
            
            # Simple list for backward compatibility
            sources = list(sources_detail.keys())
            
            return {
                "success": True,
                "documents": documents,
                "context": context,
                "sources": sources,
                "sources_detail": sources_detail,  # ← Detailed breakdown
                "count": len(documents),
                "metadatas": metadatas
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def clear_database(self) -> Dict:
        """
        Clear all documents from database
        
        Returns:
            Dictionary with result
        """
        try:
            all_ids = self.collection.get()["ids"]
            
            if all_ids:
                self.collection.delete(ids=all_ids)
                return {
                    "success": True,
                    "deleted": len(all_ids)
                }
            else:
                return {
                    "success": True,
                    "deleted": 0,
                    "message": "Database already empty"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with stats
        """
        try:
            count = self.collection.count()
            return {
                "success": True,
                "total_chunks": count,
                "collection_name": Config.COLLECTION_NAME
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }