import tiktoken
from typing import List, Dict, Tuple, Optional

class FixedTokenChunker:
    """
    Implementation of the FixedTokenChunker as described in the RAG paper.
    
    This chunker splits text into chunks of a fixed token size, with an optional
    overlap between consecutive chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 0,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: The target size of each chunk in tokens
            chunk_overlap: The number of tokens to overlap between chunks
            encoding_name: The name of the tiktoken encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def chunk_text(self, text: str) -> List[Tuple[str, Dict]]:
        """
        Split the input text into chunks.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            A list of tuples, where each tuple contains the chunk text
            and a metadata dictionary
        """
        # Encode the text into tokens
        tokens = self.encoding.encode(text)
        chunks = []
        
        # Edge case: if text is shorter than chunk_size
        if len(tokens) <= self.chunk_size:
            return [(text, {"start_idx": 0, "end_idx": len(tokens)})]
        
        # Create chunks with specified size and overlap
        start_idx = 0
        while start_idx < len(tokens):
            # Calculate end index for the current chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Decode the tokens for the current chunk back to text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create metadata for the chunk
            metadata = {
                "start_idx": start_idx,
                "end_idx": end_idx,
            }
            
            # Add the chunk to the list
            chunks.append((chunk_text, metadata))
            
            # Move the start index for the next chunk
            start_idx += self.chunk_size - self.chunk_overlap
            
            # Break if we've reached the end of the tokens
            if start_idx >= len(tokens):
                break
        
        return chunks
    
    def __call__(self, text: str) -> List[Tuple[str, Dict]]:
        """
        Make the chunker callable.
        
        Args:
            text: The text to split into chunks
            
        Returns:
            A list of tuples, where each tuple contains the chunk text
            and a metadata dictionary
        """
        return self.chunk_text(text) 