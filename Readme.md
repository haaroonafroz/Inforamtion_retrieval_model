# CV-StructRAG: Structured Information Retrieval from CVs

## Purpose

CV-StructRAG is a microservice architecture-based project that automates the extraction and structuring of information from unstructured CV PDFs. Instead of building a question-answer-based RAG pipeline, the focus here is on **information retrieval** and **intelligent segmentation** of resumes into machine-readable JSON formats.

The goal is to:
- Parse a CV PDF into identifiable sections and sub-sections (e.g., Experience → Job Title → Bullet Points).
- Convert the parsed data into a standardized nested structure.
- Support integration into frontend applications for job matching, CV scoring, or cover letter generation.
- Expose API endpoints to upload CVs, process them, and return structured JSON outputs.
- Operate locally for development and support deployment as a scalable microservice in production.

---

## Example Output Format

```json
{
  "Experience": {
    "Software Engineer - Google": [
      "Developed scalable backend services in Python",
      "Improved API performance by 30% using Redis caching"
    ],
    "Data Analyst - ABC Corp": [
      "Built dashboards in Tableau",
      "Led SQL-based data aggregation pipelines"
    ]
  },
  "Education": {
    "M.Sc. Computer Science - TU Munich": [
      "Graduated in 2022 with Distinction"
    ],
    "B.Sc. Information Technology - Delhi University": [
      "Completed in 2020 with 8.5 CGPA"
    ]
  },
  "Skills": [
    "Python", "Pandas", "TensorFlow", "Docker", "SQL"
  ]
}


## Repository Structure  
<cwd>/
├── src/
│   ├── chunking.py         # Chunking strategy to extract sections and content
│   ├── models.py           # Embedding models or section classifiers
│   ├── pipeline.py         # Full retrieval + structuring logic
│   ├── vectordb.py         # Vector database abstraction (if used)
│   ├── utils.py            # Text cleaning and formatting helpers
│   ├── config.py           # Global config for paths, model names, chunking logic
│   ├── api.py              # FastAPI app exposing backend endpoints
├── frontend/
│   └── app.py              # Streamlit app for local interaction
├── tests/
│   ├── test_chunking.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_api.py
├── Dockerfile
├── requirements.txt
└── README.md  

## Module Descriptions  

--> chunking.py:
- PDF parsing and text extraction using layout-aware tools (PyMuPDF, pdfplumber, or unstructured).

- Extracts chunks with:
    -Section title
    -Subsection/job titles (where applicable)
    -Content (bullet points, sentences)
    -Location metadata (e.g., page, block position)

- Applies heuristics and rules (e.g., heading styles, bold fonts, keywords like "Experience", "Education").

--> models.py
Loads and defines models:

- Embedding model (SentenceTransformers or similar)
- (Optional) Section classifier model to categorize chunks

- Functions:
    -embed_texts(texts: List[str]) -> np.ndarray
    -predict_section(text: str) -> str (if classification is used)

--> pipeline.py
Orchestrates the full pipeline:

- Calls chunking module to parse the PDF
- Embeds or classifies each chunk
- Groups chunks into structured dictionary format
- Outputs a final JSON with standard section headings and subsections

--> vectordb.py
If vector search is used (e.g., to retrieve matching job titles or standard section labels), this module wraps FAISS or ChromaDB logic.

- Functions:
    - add_documents(chunks)  
    - query_embedding(vector) -> ranked_chunks  
    - clear()  

--> utils.py
- Text preprocessing:
    - Normalize whitespace
    - Remove headers/footers
    - Split sentences or bullet points

- Helper functions:
    -clean_text(text)
    -detect_section_title(line)
    -merge_adjacent_chunks(...)

--> config.py
- Global parameters:
    -Chunking strategy (by_heading, by_font, by_spacing)
    -Embedding model path
    -Use_classifier: True/False
    -Default section titles
    -Logging level

--> api.py
FastAPI backend exposing endpoints:

POST /upload-pdf
  - Accepts a multipart file
  - Returns document_id or status

GET /extract-json?doc_id={id}
  - Returns parsed JSON structure

GET /health
  - Health check for deployment readiness


--> frontend/app.py
Streamlit-based interface for local dev use:

- Upload a CV PDF
- Display parsed chunks
- View final structured JSON
- Manual chunk correction support (optional)  

## Deployment
The backend can be containerized and deployed via Docker.
Possible Deployement platform option: Railway

## Extra
- Layout Aware CV-Parsing using Unstructured