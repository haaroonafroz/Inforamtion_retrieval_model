import numpy as np
from typing import List, Tuple, Dict, Callable, Any
from sklearn.metrics.pairwise import cosine_similarity
from src.chunker import FixedTokenChunker

class VectorRetriever:
    """
    A vector-based retriever that uses cosine similarity to find the most relevant chunks.
    """
    
    def __init__(
        self,
        embedding_function: Callable[[List[str]], np.ndarray],
        corpus_chunks: List[str] = None,
        corpus_embeddings: np.ndarray = None
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_function: A function that takes a list of texts and returns embeddings
            corpus_chunks: List of text chunks from the corpus (optional)
            corpus_embeddings: Precomputed embeddings for corpus chunks (optional)
        """
        self.embedding_function = embedding_function
        self.corpus_chunks = corpus_chunks or []
        self.corpus_embeddings = corpus_embeddings
    
    def index_corpus(self, corpus: List[str], chunker: FixedTokenChunker = None) -> None:
        """
        Preprocess and index a corpus for retrieval.
        
        Args:
            corpus: List of documents to index
            chunker: Chunker to use for splitting documents into chunks
        """
        if chunker is None:
            # If no chunker is provided, use the documents as-is
            self.corpus_chunks = corpus
        else:
            # Chunk each document and flatten the results
            self.corpus_chunks = []
            for doc in corpus:
                chunks = chunker(doc)
                self.corpus_chunks.extend([chunk[0] for chunk in chunks])
        
        # Generate embeddings for all chunks
        if self.corpus_chunks:
            self.corpus_embeddings = self.embedding_function(self.corpus_chunks)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: The query text
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of (chunk_text, similarity_score) tuples, sorted by decreasing similarity
        """
        if not self.corpus_chunks or self.corpus_embeddings is None:
            return []
        
        # Generate embedding for the query
        query_embedding = self.embedding_function([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_embedding, self.corpus_embeddings)[0]
        
        # Get indices of top-k chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return top-k chunks with their similarity scores
        results = [(self.corpus_chunks[i], similarities[i]) for i in top_indices]
        
        return results 