"""
Tool for managing the RAG pipeline: embedding chunks and retrieving information.
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document # For type hinting
from langchain.vectorstores.base import VectorStoreRetriever
import os
import shutil

# Configuration Defaults
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# Default to in-memory unless directory specified
DEFAULT_PERSIST_DIR = None 

class RAGManager:
    """Manages PDF chunk embedding, storage, and retrieval using ChromaDB and BGE embeddings."""
    
    def __init__(self, 
                 embedding_model_name: str | None = None, 
                 persist_directory: str | None = None,
                 embedding_device: str = 'cpu'): # Allow specifying device
        """
        Initializes the RAGManager.

        Args:
            embedding_model_name: Name of the Sentence Transformer model (e.g., BAAI/bge-base-en-v1.5).
                                  Defaults using environment variable or DEFAULT_EMBEDDING_MODEL.
            persist_directory: Path to directory for ChromaDB persistence. 
                               Defaults using environment variable or DEFAULT_PERSIST_DIR (None = in-memory).
            embedding_device: Device for embedding model ('cpu' or 'cuda').
        """
        self.embedding_model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", DEFAULT_PERSIST_DIR)
        self.embedding_device = embedding_device
        
        self.embedding_function = None
        self.vector_store = None
        print(f"RAGManager initialized. Embedding model: {self.embedding_model_name}, Persist dir: {self.persist_directory}, Device: {self.embedding_device}")

    def _get_embedding_function(self):
        """Initializes and returns the embedding function (cached)."""
        if self.embedding_function is None:
            print(f"Initializing embedding model: {self.embedding_model_name} on {self.embedding_device}")
            model_kwargs = {'device': self.embedding_device} 
            encode_kwargs = {'normalize_embeddings': True} # BGE models often require normalization
            try:
                self.embedding_function = HuggingFaceBgeEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                print("Embedding model initialized.")
            except Exception as e:
                print(f"Error initializing embedding model '{self.embedding_model_name}': {e}")
                raise # Re-raise the exception
        return self.embedding_function

    def initialize_vector_store(self, chunks: list[Document], force_recreate: bool = False):
        """
        Initializes the Chroma vector store with the given document chunks.
        Uses persistence if self.persist_directory is set.

        Args:
            chunks: A list of LangChain Document objects (the split PDF chunks).
            force_recreate: If True, deletes existing persisted data before creating.
                          Useful if the source PDF changes.
        """
        embed_func = self._get_embedding_function()
        if embed_func is None:
             print("Error: Cannot initialize vector store without embedding function.")
             return
        
        persist_path = self.persist_directory
        
        # Handle recreation if persistence is enabled
        if persist_path and force_recreate and os.path.exists(persist_path):
                print(f"Removing existing vector store at: {persist_path}")
                try:
                    shutil.rmtree(persist_path)
                except OSError as e:
                     print(f"Error removing directory {persist_path}: {e}")
                     # Decide if you want to proceed or raise

        # Create or load the vector store
        if self.vector_store is None or force_recreate:
            print("Creating new vector store...")
            try:
                if persist_path:
                    print(f"Persisting Chroma DB to: {persist_path}")
                    self.vector_store = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embed_func, 
                        persist_directory=persist_path
                    )
                else:
                    print("Creating in-memory Chroma DB.")
                    self.vector_store = Chroma.from_documents(
                        documents=chunks, 
                        embedding=embed_func
                    )
                print(f"Vector store created/loaded with {len(chunks)} chunks.")
            except Exception as e:
                print(f"Error creating Chroma vector store: {e}")
                self.vector_store = None # Ensure store is None on error
                raise
        else:
            # If not recreating and store exists, just confirm usage
            # Potentially add logic here to load from persist_path if self.vector_store is None
            # but persistence path exists (e.g., on app restart)
            print("Using existing vector store instance.")

    def get_retriever(self, k: int = 5) -> VectorStoreRetriever | None:
        """
        Gets a retriever for the initialized vector store.

        Args:
            k: The number of relevant documents to retrieve.

        Returns:
            A LangChain VectorStoreRetriever instance, or None if store not initialized.
        """
        if self.vector_store is None:
            print("Error: Vector store not initialized. Call initialize_vector_store first.")
            # Try loading if persist path exists?
            if self.persist_directory and os.path.exists(self.persist_directory):
                print(f"Attempting to load vector store from {self.persist_directory}...")
                try:
                    embed_func = self._get_embedding_function()
                    self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=embed_func)
                    print("Successfully loaded vector store from disk.")
                except Exception as e:
                    print(f"Error loading vector store from {self.persist_directory}: {e}")
                    return None
            else:
                return None # Still no store
        
        # Other retriever types could be configured here (e.g., MultiQueryRetriever)
        try:
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        except Exception as e:
             print(f"Error getting retriever: {e}")
             return None

    def query_vector_store(self, query: str, k: int = 5) -> list[Document] | None:
        """
        Queries the vector store using its retriever.
        This is the primary method to be exposed as an agent tool.

        Args:
            query: The information to search for (e.g., "User's first name").
            k: The number of relevant documents to retrieve.

        Returns:
            A list of relevant Document objects, or None if retrieval fails.
        """
        retriever = self.get_retriever(k=k)
        if retriever:
            print(f"Retrieving information for query: '{query}'")
            try:
                results = retriever.get_relevant_documents(query)
                print(f"Retrieved {len(results)} documents.")
                return results
            except Exception as e:
                 print(f"Error during retrieval for query '{query}': {e}")
                 return None
        else:
            print("Retrieval failed: Could not get retriever.")
            return None

# Example Usage (for testing - requires load_and_process_pdf.py)
if __name__ == '__main__':
    # Assume load_and_process_pdf.py is in the same directory or PYTHONPATH
    try:
        from load_and_process_pdf import load_and_split_pdf
        dummy_pdf_path = "example_cv.pdf" # Use the same test PDF
        
        if os.path.exists(dummy_pdf_path):
            print("--- Testing RAG Pipeline --- RAGManager Class ---")
            
            # 1. Instantiate the manager (using defaults or env vars)
            # Set force_recreate_on_init = True if you always want a fresh DB for this test run
            force_recreate_on_init = True 
            rag_manager = RAGManager()

            # 2. Load and split PDF
            print("Loading and splitting PDF...")
            chunks = load_and_split_pdf(dummy_pdf_path)
            
            # 3. Initialize vector store using the manager instance
            print("Initializing vector store...")
            rag_manager.initialize_vector_store(chunks, force_recreate=force_recreate_on_init)
            
            # 4. Perform a query using the manager instance
            test_query = "What is the candidate's work experience?"
            retrieved_docs = rag_manager.query_vector_store(test_query, k=3)
            
            if retrieved_docs:
                print(f"\n--- Results for query: '{test_query}' ---")
                for i, doc in enumerate(retrieved_docs):
                    print(f"Result {i+1}:\nSource: {doc.metadata.get('source', 'N/A')}, Chunk: {doc.metadata.get('chunk_index', 'N/A')}\nContent: {doc.page_content[:200]}...\n") # Print snippet
            elif rag_manager.vector_store is not None: # Check if query failed vs store not init
                print(f"Query '{test_query}' did not return results from the vector store.")
            else:
                 print("Query failed because vector store could not be initialized or loaded.")
                 
            # Test loading from persistence if applicable
            if rag_manager.persist_directory:
                print("\n--- Testing Loading from Persistence ---")
                rag_manager_reloaded = RAGManager(persist_directory=rag_manager.persist_directory)
                # Don't initialize, just query (should load automatically in get_retriever)
                retrieved_docs_reloaded = rag_manager_reloaded.query_vector_store(test_query, k=3)
                if retrieved_docs_reloaded:
                    print(f"Successfully retrieved {len(retrieved_docs_reloaded)} docs after reloading manager.")
                else:
                    print("Failed to retrieve docs after reloading manager.")
                 
        else:
            print(f"Test PDF '{dummy_pdf_path}' not found. Skipping example usage.")
            
    except ImportError:
        print("Could not import load_and_split_pdf. Make sure it's accessible.")
    except Exception as e:
        import traceback
        print(f"Testing failed: {e}")
        traceback.print_exc() # Print full traceback for debugging 