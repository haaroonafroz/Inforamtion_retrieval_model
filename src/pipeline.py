import os
import logging
from typing import List, Dict, Any, Optional, Union
import tempfile
from pathlib import Path

from src.chunking import ResumeChunker
from src.models import create_embedding_model, EmbeddingModel, SectionClassifier
from src.vectordb import create_vector_store, VectorStore
from src.utils import save_json, load_json, clean_text, extract_contact_info

# Configure logging
logger = logging.getLogger(__name__)

class CVProcessingPipeline:
    """
    Main pipeline for processing CV/Resume PDFs
    """
    
    def __init__(self, 
                config: Optional[Dict[str, Any]] = None, 
                embedding_model_type: str = "sentence_transformer",
                vector_store_type: str = "chroma",
                use_classifier: bool = False,
                layout_model_name: str = "microsoft/layoutlmv3-base"):
        """
        Initialize the CV processing pipeline
        
        Args:
            config: Optional configuration dictionary
            embedding_model_type: Type of embedding model to use
            vector_store_type: Type of vector store to use
            use_classifier: Whether to use section classifier
            layout_model_name: Name of the LayoutLM model to use (passed to chunker)
        """
        self.config = config or {}
        
        # Set up output directory
        self.output_dir = self.config.get("output_directory", "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize chunker with the layout-aware capabilities
        logger.info(f"Initializing resume chunker with layout model: {layout_model_name}")
        self.chunker = ResumeChunker(
            model_name=layout_model_name or self.config.get("layout_model_name", "microsoft/layoutlmv3-base")
        )
        
        # Initialize embedding model
        # Determine model name based on embedding model type
        if embedding_model_type == "huggingface_api":
            model_name = self.config.get("embedding_model_bge", self.config.get("embedding_model", "BAAI/bge-large-en-v1.5"))
        elif embedding_model_type == "custom":
            model_name = self.config.get("embedding_model_e5large", self.config.get("embedding_model", "intfloat/e5-large-v2"))
        else:  # sentence_transformer
            model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        
        logger.info(f"Initializing embedding model: {embedding_model_type} - {model_name}")
        self.embedding_model = create_embedding_model(
            model_type=embedding_model_type,
            model_name=model_name
        )
        
        # Initialize vector store
        persist_dir = self.config.get("vector_store_directory", os.path.join(self.output_dir, "vector_store"))
        logger.info(f"Initializing vector store: {vector_store_type}")
        self.vector_store = create_vector_store(
            embedding_model=self.embedding_model,
            store_type=vector_store_type,
            collection_name="cv_sections",
            persist_directory=persist_dir
        )
        
        # Initialize classifier if needed
        self.use_classifier = use_classifier
        if use_classifier:
            logger.info("Initializing section classifier")
            self.classifier = SectionClassifier()
        else:
            self.classifier = None
    
    def process_pdf(self, pdf_path: str, save_output: bool = True) -> Dict[str, Any]:
        """
        Process a single PDF file
        
        Args:
            pdf_path: Path to the PDF file
            save_output: Whether to save the output to disk
            
        Returns:
            Dictionary containing the processed data
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Process the PDF using the chunker
            result = self.chunker.process_pdf(pdf_path)
            
            # Extract hierarchical and flat chunks
            hierarchical_chunks = result["hierarchical_chunks"]
            flat_chunks = result["flat_chunks"]
            structured_data = result["output_json"]
            
            # Add chunks to vector store if needed
            if self.config.get("populate_vector_store", True):
                logger.info(f"Adding {len(flat_chunks)} chunks to vector store")
                self.vector_store.add_documents(flat_chunks)
            
            # Classify sections if needed
            if self.use_classifier:
                logger.info("Classifying sections")
                for chunk in flat_chunks:
                    if chunk["metadata"]["type"] == "section":
                        predicted_section = self.classifier.predict_section(chunk["text"])
                        chunk["metadata"]["classified_heading"] = predicted_section
            
            # Prepare the result
            result = {
                "hierarchical_chunks": hierarchical_chunks,
                "flat_chunks": flat_chunks,
                "structured_data": structured_data,
                "metadata": {
                    "source_file": os.path.basename(pdf_path),
                    "processing_time": None  # Could add timing info here
                }
            }
            
            # Save results if needed
            if save_output:
                output_file = os.path.join(
                    self.output_dir, 
                    f"{Path(pdf_path).stem}_processed.json"
                )
                save_json(structured_data, output_file)
                logger.info(f"Saved structured output to {output_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            
            # Return an empty but correctly structured result to avoid downstream errors
            return {
                "hierarchical_chunks": {},
                "flat_chunks": [],
                "structured_data": {
                    "personalInfo": {},
                    "experience": [],
                    "education": [],
                    "skills": [],
                    "projects": [],
                    "summary": ""
                },
                "metadata": {
                    "source_file": os.path.basename(pdf_path),
                    "processing_time": None,
                    "error": str(e)
                }
            }
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of processing results
        """
        logger.info(f"Processing directory: {directory_path}")
        
        results = []
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            try:
                result = self.process_pdf(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
        
        return results
    
    def search_sections(self, query: str, section_filter: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant sections using the vector store
        
        Args:
            query: The search query
            section_filter: Optional section type to filter by
            k: Number of results to return
            
        Returns:
            List of matching sections
        """
        logger.info(f"Searching for: {query}")
        
        # Build filter if section is specified
        filter_metadata = {}
        if section_filter:
            filter_metadata["normalized_heading"] = section_filter
        
        # Perform search
        results = self.vector_store.search(
            query=query,
            k=k,
            filter_metadata=filter_metadata if filter_metadata else None
        )
        
        return results
    
    def extract_structured_data(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract structured data from PDF without using vector store
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Structured data dictionary
        """
        logger.info(f"Extracting structured data from: {pdf_path}")
        
        # Process PDF
        chunks_data = self.chunker.process_pdf(pdf_path)
        
        # Return only the structured output
        return chunks_data["output_json"]
    
    def classify_sections(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify sections in chunks if classifier is available
        
        Args:
            chunks: List of chunks to classify
            
        Returns:
            Classified chunks
        """
        if not self.use_classifier or not self.classifier:
            logger.warning("No classifier available, skipping classification")
            return chunks
        
        logger.info(f"Classifying {len(chunks)} chunks")
        
        for chunk in chunks:
            if chunk["metadata"]["type"] == "section":
                predicted_section = self.classifier.predict_section(chunk["text"])
                chunk["metadata"]["classified_heading"] = predicted_section
        
        return chunks
    
    def clear_vector_store(self) -> None:
        """Clear the vector store"""
        logger.info("Clearing vector store")
        self.vector_store.clear()


# Example usage
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create pipeline
    pipeline = CVProcessingPipeline()
    
    # Process a PDF if provided as argument
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if os.path.isfile(pdf_path) and pdf_path.lower().endswith('.pdf'):
            result = pipeline.process_pdf(pdf_path)
            print(f"Processed {pdf_path}")
            print(f"Found {len(result['structured_data'])} sections")
        elif os.path.isdir(pdf_path):
            results = pipeline.process_directory(pdf_path)
            print(f"Processed {len(results)} PDFs in {pdf_path}")
        else:
            print(f"Invalid path: {pdf_path}")
    else:
        print("Usage: python -m src.pipeline <pdf_path_or_directory>")
