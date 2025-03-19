import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """
    A class for generating embeddings from text using a SentenceTransformer model.
    
    Supported models:
    - "all-MiniLM-L6-v2"
    - "multi-qa-mpnet-base-dot-v1"
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: The name of the model to use for embeddings
            device: The device to use for computation ('cpu', 'cuda', etc.)
                   If None, will use CUDA if available, else CPU
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
    
    def embed_texts(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize the embeddings to unit length
            
        Returns:
            A numpy array of embeddings, shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=normalize
        )
        return embeddings
    
    def __call__(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Make the embedding model callable.
        
        Args:
            texts: Text or list of texts to embed
            normalize: Whether to normalize the embeddings to unit length
            
        Returns:
            A numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        return self.embed_texts(texts, normalize) 