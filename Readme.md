# RAG Retrieval Pipeline Evaluation

This project implements a retrieval-augmented generation (RAG) pipeline using open-source embedding models and evaluates the retrieval quality on different datasets. The implementation includes a fixed token chunking algorithm, embedding generation, and evaluation metrics for precision and recall.

## Project Structure

```
.
├── data/
│   ├── corpora/       # Text corpora for evaluation
│   ├── questions_df.csv  # Queries and relevant excerpts
│   └── results/       # Evaluation results
├── src/
│   ├── chunker.py       # FixedTokenChunker implementation
│   ├── embeddings.py    # Embedding model wrapper
│   ├── metrics.py       # Precision and Recall metrics
│   ├── retriever.py     # Vector-based retriever
│   ├── evaluation.py    # Evaluation pipeline
│   └── main.py          # Main script to run evaluation
├── download_datasets.py  # Script to download datasets
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the datasets:

```bash
python download_datasets.py
```

## Usage

Run the main evaluation script:

```bash
# Basic usage with default parameters
python -m src.main

# Customized parameters
python -m src.main --dataset wikitext --model_name all-MiniLM-L6-v2 --chunk_sizes 200 400 --top_k_values 5 10
```

### Command-line Arguments

- `--dataset`: Dataset to use for evaluation (choices: "wikitext", "chatlogs", "state_of_the_union", default: "wikitext")
- `--corpus_dir`: Directory containing corpus documents (default: "data/corpora")
- `--questions_path`: Path to the questions CSV file (default: "data/questions_df.csv")
- `--model_name`: Embedding model to use (choices: "all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1", default: "all-MiniLM-L6-v2")
- `--chunk_sizes`: Chunk sizes to evaluate (default: 200 400)
- `--top_k_values`: Number of retrieved chunks to evaluate (default: 5 10)
- `--results_dir`: Directory to save results (default: "data/results")

## Evaluation Results

The evaluation results are saved in two formats:

1. Detailed JSON files for each experiment configuration:
   - `data/results/{dataset}_{model_name}_chunk{chunk_size}.json`

2. Summary CSV file with all configurations:
   - `data/results/{dataset}_{model_name}_summary.csv`

The summary table includes the following metrics for each configuration:
- Precision
- Recall
- F1 Score
