import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import json

from src.chunker import FixedTokenChunker
from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever
from src.metrics import precision_at_k, recall_at_k, f1_score

class RetrievalEvaluator:
    """
    Evaluator for retrieval quality using precision and recall metrics.
    """
    
    def __init__(
        self,
        corpus_docs: List[str],
        queries: List[str],
        relevant_chunks: List[List[str]],
        chunker: FixedTokenChunker,
        embedding_model: EmbeddingModel
    ):
        """
        Initialize the evaluator.
        
        Args:
            corpus_docs: List of documents in the corpus
            queries: List of query texts
            relevant_chunks: List of lists of relevant chunks for each query
            chunker: Chunker to use for splitting documents
            embedding_model: Model to use for generating embeddings
        """
        self.corpus_docs = corpus_docs
        self.queries = queries
        self.relevant_chunks = relevant_chunks
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.retriever = VectorRetriever(embedding_model)
        
        # Index the corpus
        print(f"Indexing corpus with {len(corpus_docs)} documents...")
        self.retriever.index_corpus(corpus_docs, chunker)
        print(f"Indexed {len(self.retriever.corpus_chunks)} chunks")
    
    def evaluate(self, top_k_values: List[int] = [5, 10]) -> Dict[str, Any]:
        """
        Evaluate retrieval quality for different values of k.
        
        Args:
            top_k_values: List of k values to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        for k in top_k_values:
            precisions = []
            recalls = []
            f1_scores = []
            
            print(f"Evaluating with top_k={k}")
            for i, query in enumerate(tqdm(self.queries)):
                print(f"\nQuery {i+1}: '{query}'")
                print(f"Expected relevant text: '{self.relevant_chunks[i]}'")
                
                # Retrieve chunks
                retrieved_chunks_with_scores = self.retriever.retrieve(query, top_k=k)
                retrieved_chunks = [chunk for chunk, _ in retrieved_chunks_with_scores]
                
                print(f"Retrieved top {len(retrieved_chunks)} chunks:")
                for j, (chunk, score) in enumerate(retrieved_chunks_with_scores[:3]):  # Show only top 3 for brevity
                    print(f"  {j+1}. (score={score:.4f}): '{chunk[:100]}...'")
                
                # Calculate metrics
                precision = precision_at_k(retrieved_chunks, self.relevant_chunks[i], k)
                recall = recall_at_k(retrieved_chunks, self.relevant_chunks[i], k)
                f1 = f1_score(precision, recall)
                
                print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
            
            # Calculate average metrics
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1_scores)
            
            print(f"\nAverage for top_{k}:")
            print(f"  Precision: {avg_precision}")
            print(f"  Recall: {avg_recall}")
            print(f"  F1: {avg_f1}")
            
            results[f"top_{k}"] = {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "individual_results": {
                    "precisions": precisions,
                    "recalls": recalls,
                    "f1_scores": f1_scores
                }
            }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Dictionary with evaluation results
            output_path: Path to save the results file
        """
        # Convert numpy arrays to Python lists for JSON serialization
        json_results = {}
        
        for k, metrics in results.items():
            json_results[k] = {
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
            }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

def prepare_dataset(
    corpus_dir: str, 
    questions_path: str, 
    dataset_name: str = "wikitext"
) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    Prepare the dataset for evaluation.
    
    Args:
        corpus_dir: Directory containing corpus documents
        questions_path: Path to the questions CSV file
        dataset_name: Name of the dataset to use
        
    Returns:
        Tuple of (corpus_docs, queries, relevant_chunks)
    """
    # Load corpus documents
    corpus_docs = []
    corpus_files = [f for f in os.listdir(corpus_dir) if dataset_name.lower() in f.lower()]
    
    print(f"Loading {len(corpus_files)} corpus files for {dataset_name}...")
    for file_name in corpus_files:
        with open(os.path.join(corpus_dir, file_name), 'r', encoding='utf-8') as f:
            corpus_docs.append(f.read())
    
    # Load questions and relevant chunks
    questions_df = pd.read_csv(questions_path)
    
    # Filter questions for the selected dataset
    questions_df = questions_df[questions_df['dataset'] == dataset_name]
    
    print(f"Loaded {len(questions_df)} questions for {dataset_name}")
    
    queries = questions_df['question'].tolist()
    relevant_chunks = questions_df['golden_text'].apply(lambda x: [x]).tolist()
    
    return corpus_docs, queries, relevant_chunks 