from typing import List, Set
import numpy as np

def precision_at_k(retrieved_chunks: List[str], relevant_chunks: List[str], k: int = None) -> float:
    """
    Calculate precision@k for retrieved chunks.
    
    Precision@k = (# of relevant chunks in top-k retrieved) / k
    
    Args:
        retrieved_chunks: List of retrieved chunks
        relevant_chunks: List of relevant (golden) chunks
        k: Number of top chunks to consider. If None, use all retrieved chunks.
        
    Returns:
        Precision@k score (between 0 and 1)
    """
    if k is None:
        k = len(retrieved_chunks)
    else:
        k = min(k, len(retrieved_chunks))
    
    if k == 0:
        return 0.0
    
    # Consider only the top-k retrieved chunks
    top_k_chunks = retrieved_chunks[:k]
    
    # Check if any retrieved chunk contains the relevant text
    relevant_retrieved = 0
    for retrieved in top_k_chunks:
        for relevant in relevant_chunks:
            if relevant.lower() in retrieved.lower():
                relevant_retrieved += 1
                break  # Count this retrieved chunk only once
    
    # Calculate precision
    precision = relevant_retrieved / k
    
    return precision

def recall_at_k(retrieved_chunks: List[str], relevant_chunks: List[str], k: int = None) -> float:
    """
    Calculate recall@k for retrieved chunks.
    
    Recall@k = (# of relevant chunks in top-k retrieved) / (total # of relevant chunks)
    
    Args:
        retrieved_chunks: List of retrieved chunks
        relevant_chunks: List of relevant (golden) chunks
        k: Number of top chunks to consider. If None, use all retrieved chunks.
        
    Returns:
        Recall@k score (between 0 and 1)
    """
    if not relevant_chunks:
        return 1.0  # If there are no relevant chunks, recall is perfect
    
    if k is None:
        k = len(retrieved_chunks)
    else:
        k = min(k, len(retrieved_chunks))
    
    if k == 0:
        return 0.0
    
    # Consider only the top-k retrieved chunks
    top_k_chunks = retrieved_chunks[:k]
    
    # Check which relevant chunks are found in the retrieved chunks
    relevant_found = set()
    for i, relevant in enumerate(relevant_chunks):
        for retrieved in top_k_chunks:
            if relevant.lower() in retrieved.lower():
                relevant_found.add(i)
                break
    
    # Calculate recall
    recall = len(relevant_found) / len(relevant_chunks)
    
    return recall

def f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score (between 0 and 1)
    """
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1 