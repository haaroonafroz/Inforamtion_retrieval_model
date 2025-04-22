import os
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from models import EmbeddingModel

# Try to import vector database libraries
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class VectorStore:
    """Base class for vector databases"""
    
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents similar to the query"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def clear(self):
        """Clear all documents from the vector store"""
        raise NotImplementedError("Subclasses must implement this method")


class ChromaVectorStore(VectorStore):
    """Vector store using ChromaDB"""
    
    def __init__(self, 
                embedding_model: EmbeddingModel, 
                collection_name: str = "resume_sections", 
                persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB vector store
        
        Args:
            embedding_model: Model to use for embeddings
            collection_name: Name of the collection in ChromaDB
            persist_directory: Directory to persist the database to
        """
        super().__init__(embedding_model)
        
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Please install it with: pip install chromadb")
        
        logger.info(f"Initializing ChromaDB vector store with collection: {collection_name}")
        
        # Initialize client
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents to add. Each document should have:
                - text: The document text
                - metadata: Optional metadata
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to ChromaDB collection")
        
        # Extract document components
        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Add documents in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            logger.debug(f"Adding batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            batch_ids = ids[i:end_idx]
            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB collection")
    
    def search(self, query: str, k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query
        
        Args:
            query: The search query
            k: Number of results to return
            filter_metadata: Optional filter to apply on metadata
        
        Returns:
            List of documents with scores
        """
        logger.debug(f"Searching for: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Build where clause if filter_metadata is provided
        where = None
        if filter_metadata:
            where = {}
            for key, value in filter_metadata.items():
                where[key] = value
        
        # Perform search
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where
            )
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return []
        
        # Format results
        formatted_results = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                
                # Convert distance to score (1 - distance for cosine)
                score = 1.0 - distance
                
                formatted_results.append({
                    "text": doc,
                    "metadata": metadata,
                    "score": score
                })
        
        return formatted_results
    
    def clear(self):
        """Clear all documents from the collection"""
        logger.info(f"Clearing collection: {self.collection_name}")
        self.collection.delete(where={})


class FaissVectorStore(VectorStore):
    """Vector store using FAISS"""
    
    def __init__(self, embedding_model: EmbeddingModel, index_path: Optional[str] = None):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_model: Model to use for embeddings
            index_path: Path to save/load the FAISS index
        """
        super().__init__(embedding_model)
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Please install it with: pip install faiss-cpu or faiss-gpu")
        
        logger.info("Initializing FAISS vector store")
        
        # Initialize index
        self.dimension = embedding_model.dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity with normalized vectors
        
        # Load index if it exists
        if index_path and os.path.exists(index_path):
            logger.info(f"Loading existing FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
        
        self.index_path = index_path
        self.documents = []  # Store document texts and metadata
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents to add. Each document should have:
                - text: The document text
                - metadata: Optional metadata
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to FAISS index")
        
        # Generate embeddings
        texts = [doc["text"] for doc in documents]
        embeddings = np.array(self.embedding_model.embed_documents(texts), dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store document texts and metadata
        for doc in documents:
            self.documents.append({
                "text": doc["text"],
                "metadata": doc.get("metadata", {})
            })
        
        # Save index if path is provided
        if self.index_path:
            logger.info(f"Saving FAISS index to {self.index_path}")
            faiss.write_index(self.index, self.index_path)
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    def search(self, query: str, k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query
        
        Args:
            query: The search query
            k: Number of results to return
            filter_metadata: Optional filter to apply on metadata
        
        Returns:
            List of documents with scores
        """
        logger.debug(f"Searching for: {query[:50]}...")
        
        if self.index.ntotal == 0:
            logger.warning("Index is empty, no results to return")
            return []
        
        # Generate query embedding
        query_embedding = np.array([self.embedding_model.embed_query(query)], dtype=np.float32)
        
        # Normalize embedding for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            score = float(scores[0][i])
            
            # Apply metadata filter if provided
            if filter_metadata:
                metadata = doc.get("metadata", {})
                skip = False
                for key, value in filter_metadata.items():
                    if key not in metadata or metadata[key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            results.append({
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "score": score
            })
        
        return results
    
    def clear(self):
        """Clear the vector store"""
        logger.info("Clearing FAISS index")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        
        # Save empty index if path is provided
        if self.index_path:
            faiss.write_index(self.index, self.index_path)


# Factory function to create vector store
def create_vector_store(embedding_model: EmbeddingModel, 
                        store_type: str = "chroma",
                        **kwargs) -> VectorStore:
    """
    Factory function to create a vector store
    
    Args:
        embedding_model: The embedding model to use
        store_type: Type of vector store ("chroma", "faiss")
        **kwargs: Additional arguments for the specific store
    
    Returns:
        An instance of VectorStore
    """
    logger.info(f"Creating vector store of type: {store_type}")
    
    if store_type == "chroma":
        if not CHROMA_AVAILABLE:
            logger.error("ChromaDB is not installed. Please install it with: pip install chromadb")
            raise ImportError("ChromaDB is not installed. Please install it with: pip install chromadb")
            
        collection_name = kwargs.get("collection_name", "resume_sections")
        persist_directory = kwargs.get("persist_directory", None)
        
        return ChromaVectorStore(
            embedding_model=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    
    elif store_type == "faiss":
        if not FAISS_AVAILABLE:
            logger.error("FAISS is not installed. Please install it with: pip install faiss-cpu or faiss-gpu")
            raise ImportError("FAISS is not installed. Please install it with: pip install faiss-cpu or faiss-gpu")
            
        index_path = kwargs.get("index_path", None)
        
        return FaissVectorStore(
            embedding_model=embedding_model,
            index_path=index_path
        )
    
    else:
        error_msg = f"Unknown vector store type: {store_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)


# Example usage
if __name__ == "__main__":
    # Test vector store with a simple example
    from models import create_embedding_model
    
    # Create embedding model
    model = create_embedding_model("sentence_transformer")
    
    # Create vector store
    vector_store = create_vector_store(
        embedding_model=model,
        store_type="chroma",
        collection_name="test_collection",
        persist_directory="./chroma_db"
    )
    
    # Add documents
    test_docs = [
        {"text": "This is a test document about Python programming", 
         "metadata": {"type": "section", "heading": "Skills"}},
        {"text": "Another document about machine learning and AI", 
         "metadata": {"type": "section", "heading": "Skills"}},
        {"text": "Document about work experience as a developer", 
         "metadata": {"type": "section", "heading": "Experience"}}
    ]
    
    vector_store.add_documents(test_docs)
    
    # Search
    results = vector_store.search("Python programming skills")
    
    print("Search results:")
    for result in results:
        print(f"Score: {result['score']:.4f}, Text: {result['text'][:50]}...")
