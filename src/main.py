import os
import argparse
import json
from tqdm import tqdm
import pandas as pd

from src.chunker import FixedTokenChunker
from src.embeddings import EmbeddingModel
from src.evaluation import RetrievalEvaluator, prepare_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline")
    parser.add_argument("--dataset", type=str, default="wikitext", 
                        choices=["wikitext", "chatlogs", "state_of_the_union"],
                        help="Dataset to use for evaluation")
    parser.add_argument("--corpus_dir", type=str, default="data/corpora", 
                        help="Directory containing corpus documents")
    parser.add_argument("--questions_path", type=str, default="data/questions_df.csv",
                        help="Path to the questions CSV file")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                        choices=["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1"],
                        help="Embedding model to use")
    parser.add_argument("--chunk_sizes", type=int, nargs="+", default=[200, 400],
                        help="Chunk sizes to evaluate")
    parser.add_argument("--top_k_values", type=int, nargs="+", default=[5, 10],
                        help="Number of retrieved chunks to evaluate")
    parser.add_argument("--results_dir", type=str, default="data/results",
                        help="Directory to save results")
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Prepare dataset
    print(f"Preparing dataset: {args.dataset}")
    corpus_docs, queries, relevant_chunks = prepare_dataset(
        args.corpus_dir, args.questions_path, args.dataset
    )
    
    # Initialize embedding model
    print(f"Initializing embedding model: {args.model_name}")
    embedding_model = EmbeddingModel(args.model_name)
    
    # Initialize results table
    results_table = []
    
    # Run experiments with different chunking parameters
    for chunk_size in args.chunk_sizes:
        print(f"Evaluating with chunk size: {chunk_size}")
        
        # Initialize chunker
        chunker = FixedTokenChunker(chunk_size=chunk_size)
        
        # Initialize evaluator
        evaluator = RetrievalEvaluator(
            corpus_docs=corpus_docs,
            queries=queries,
            relevant_chunks=relevant_chunks,
            chunker=chunker,
            embedding_model=embedding_model
        )
        
        # Run evaluation
        results = evaluator.evaluate(top_k_values=args.top_k_values)
        
        # Save detailed results
        output_path = os.path.join(
            args.results_dir, 
            f"{args.dataset}_{args.model_name}_chunk{chunk_size}.json"
        )
        evaluator.save_results(results, output_path)
        
        # Add to results table
        for k in args.top_k_values:
            results_table.append({
                "dataset": args.dataset,
                "model": args.model_name,
                "chunk_size": chunk_size,
                "top_k": k,
                "precision": results[f"top_{k}"]["precision"],
                "recall": results[f"top_{k}"]["recall"],
                "f1": results[f"top_{k}"]["f1"]
            })
    
    # Save results table as CSV
    results_df = pd.DataFrame(results_table)
    results_df.to_csv(
        os.path.join(args.results_dir, f"{args.dataset}_{args.model_name}_summary.csv"),
        index=False
    )
    
    # Print summary table
    print("\nResults Summary:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main() 