import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from src.layout_chunking import LayoutAwareResumeChunker
from src.pipeline import CVProcessingPipeline
from src.config import load_config

def process_pdf_with_layout(pdf_path, output_dir="output", save_json=True):
    """
    Process a PDF using the LayoutAwareResumeChunker directly
    """
    logger.info(f"Processing PDF with LayoutAwareResumeChunker: {pdf_path}")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the chunker
    chunker = LayoutAwareResumeChunker()
    
    # Process the PDF
    result = chunker.process_pdf(pdf_path)
    
    # Save result if needed
    if save_json:
        output_file = os.path.join(
            output_dir,
            f"{Path(pdf_path).stem}_layout.json"
        )
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved layout processing result to {output_file}")
    
    return result

def process_pdf_with_pipeline(pdf_path, output_dir="output"):
    """
    Process a PDF using the CVProcessingPipeline with layout awareness
    """
    logger.info(f"Processing PDF with layout-aware pipeline: {pdf_path}")
    
    # Load config
    config = load_config()
    
    # Create the pipeline with layout-aware processing
    pipeline = CVProcessingPipeline(
        config=config,
        use_layout_aware=True
    )
    
    # Process the PDF
    result = pipeline.process_pdf(pdf_path)
    
    return result

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Test LayoutAwareResumeChunker")
    parser.add_argument(
        "--pdf", 
        required=True, 
        help="Path to the PDF file to process"
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Directory to save output (default: 'output')"
    )
    parser.add_argument(
        "--method",
        choices=["direct", "pipeline", "both"],
        default="both",
        help="Processing method to use: direct, pipeline, or both (default: both)"
    )
    
    args = parser.parse_args()
    
    # Check if PDF exists
    if not os.path.exists(args.pdf):
        logger.error(f"PDF file not found: {args.pdf}")
        return
    
    # Process PDF with the chosen method
    if args.method in ["direct", "both"]:
        logger.info("Using direct LayoutAwareResumeChunker:")
        try:
            result_direct = process_pdf_with_layout(args.pdf, args.output)
            logger.info("Direct processing completed successfully")
        except Exception as e:
            logger.error(f"Error in direct processing: {str(e)}")
    
    if args.method in ["pipeline", "both"]:
        logger.info("Using CVProcessingPipeline with layout-aware option:")
        try:
            result_pipeline = process_pdf_with_pipeline(args.pdf, args.output)
            logger.info("Pipeline processing completed successfully")
        except Exception as e:
            logger.error(f"Error in pipeline processing: {str(e)}")

if __name__ == "__main__":
    main() 