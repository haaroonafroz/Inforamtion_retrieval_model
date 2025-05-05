"""
Tool for loading a PDF file and splitting it into text chunks.
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Consider making chunk_size and chunk_overlap configurable
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

def load_and_split_pdf(pdf_file_path: str, 
                       chunk_size: int = DEFAULT_CHUNK_SIZE, 
                       chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    """
    Loads a PDF using PyPDFLoader and splits it using RecursiveCharacterTextSplitter.

    Args:
        pdf_file_path: Path to the PDF file.
        chunk_size: Maximum size of chunks.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A list of LangChain Document objects (chunks).
    """
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_file_path}")

    try:
        # 1. Load the PDF
        loader = PyPDFLoader(pdf_file_path)
        # Loads pages as separate documents by default
        pages = loader.load() 
        
        # Combine page texts if you want to split across pages, 
        # or process page by page if structure is page-dependent.
        # For CVs, combining might be better initially.
        full_text = "\n".join([page.page_content for page in pages])
        
        # Re-create a single document source for splitting (optional but cleaner)
        # You might want to retain original page metadata if processing page-by-page
        # For now, we treat the whole PDF as one source for splitting.
        
        # 2. Initialize the splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Common separators, add more if needed for specific CV formats
            separators=["\n\n", "\n", ". ", " ", ""] 
        )

        # 3. Split the text
        chunks = text_splitter.create_documents([full_text])
        
        # Optional: Add metadata back to chunks if needed (e.g., source file name)
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = pdf_file_path
            chunk.metadata["chunk_index"] = i
            # Add other metadata if useful (e.g., derive from page content if not combined)

        print(f"Successfully loaded and split '{pdf_file_path}' into {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        print(f"Error processing PDF {pdf_file_path}: {e}")
        # Depending on agent design, might want to raise e or return None/[]
        raise 

# Example usage (for testing)
if __name__ == '__main__':
    # Create a dummy PDF or replace with a real path for testing
    dummy_pdf_path = "example_cv.pdf" 
    # Ensure you have a PDF at this path if running directly
    if os.path.exists(dummy_pdf_path):
        try:
            document_chunks = load_and_split_pdf(dummy_pdf_path)
            print(f"First chunk:\n{document_chunks[0].page_content}\n...")
            print(f"Metadata of first chunk: {document_chunks[0].metadata}")
        except Exception as e:
            print(f"Testing failed: {e}")
    else:
        print(f"Test PDF '{dummy_pdf_path}' not found. Skipping example usage.") 