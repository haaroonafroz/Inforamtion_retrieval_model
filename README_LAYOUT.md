# Layout-Aware CV Processing

This module provides layout-aware CV/resume processing capabilities using LayoutLMv3, a document understanding model that understands both text content and visual layout.

## Features

- **Layout-Aware Text Extraction**: Uses pdfplumber to extract text with font size, style, and position information
- **Document Structure Recognition**: Identifies headings, subheadings, and content based on visual layout cues
- **Hierarchical Information Extraction**: Organizes CV/resume content into a structured format preserving relationships
- **Integration with Existing Pipeline**: Seamlessly integrates with the current CV processing pipeline

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- pdfplumber
- Pillow
- OpenCV
- pytesseract (optional, for OCR capabilities)

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. If using OCR capabilities on Linux/Ubuntu, install Tesseract:

```bash
apt-get update && apt-get install -y tesseract-ocr libtesseract-dev
```

## Usage

### Direct Usage

```python
from src.layout_chunking import LayoutAwareResumeChunker

# Initialize the chunker
chunker = LayoutAwareResumeChunker()

# Process a PDF
result = chunker.process_pdf("path/to/resume.pdf")

# The result is a structured JSON object containing extracted CV data
print(result)
```

### Using the CV Processing Pipeline

```python
from src.pipeline import CVProcessingPipeline
from src.config import load_config

# Load configuration
config = load_config()

# Create pipeline with layout-aware processing
pipeline = CVProcessingPipeline(
    config=config,
    use_layout_aware=True
)

# Process a PDF
result = pipeline.process_pdf("path/to/resume.pdf")
```

### Using the Test Script

```bash
python test_layout_chunker.py --pdf path/to/resume.pdf --output output_dir
```

### Using the Streamlit App

1. Run the Streamlit app:

```bash
streamlit run frontend/app.py
```

2. In the sidebar, check the "Use Layout-Aware Processing" option
3. Upload a CV/Resume PDF
4. View the structured output

## How It Works

1. **Text Extraction**: The PDF is processed using pdfplumber to extract words along with their font information, size, and bounding box coordinates.

2. **Line Formation**: Words are grouped into lines based on their vertical position on the page.

3. **Line Classification**: Each line is classified as a heading, subheading, or content based on:
   - Font size relative to the average font size in the document
   - Bold formatting
   - Content patterns (section headings, job titles, etc.)
   - Positional context within the document

4. **Hierarchical Structure Building**: Classified lines are organized into a hierarchical structure:
   - Headings become main sections
   - Subheadings become subsections
   - Content is attached to the appropriate section/subsection

5. **Structured JSON Creation**: The hierarchical structure is converted to a standardized JSON format with specific fields for different types of information:
   - Personal information (name, contact details)
   - Experience
   - Education
   - Skills
   - Projects
   - etc.

## Output Format

The output is a structured JSON object with the following main sections:

- `personalInfo`: Name, email, phone, location, social links
- `experience`: List of work experiences with title, company, dates, and responsibilities
- `education`: List of education entries with degree, institution, dates
- `skills`: List of skills
- `projects`: List of projects with details
- `summary`: Professional summary or objective

## Limitations

- This approach relies heavily on visual formatting cues and may not work as well for unusually formatted CVs/resumes
- For best results, the PDF should be properly digitized with embedded text (not scanned images)
- Large documents may require more processing time, especially on first run when downloading the model

## Future Improvements

- Fine-tuning LayoutLMv3 specifically for resume parsing
- Implementing OCR for scanned documents
- Adding support for more languages
- Improving extraction of dates and complex formatting 