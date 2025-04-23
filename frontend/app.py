import os
import sys
import json
import tempfile
import streamlit as st
import pandas as pd
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import CVProcessingPipeline
from src.config import load_config
from src.models import create_embedding_model
from src.vectordb import create_vector_store

# Page configuration
st.set_page_config(
    page_title="CV-StructRAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize pipeline
@st.cache_resource(show_spinner=False)
def get_pipeline(model_selection="all-MiniLM", use_classifier=False, _version=1):
    """
    Create a CV processing pipeline with the selected model
    
    Args:
        model_selection: Which embedding model to use
        use_classifier: Whether to use the section classifier
        _version: Cache version (increment to force rebuild)
    """
    config = load_config()
    
    # Choose the appropriate model based on selection
    if model_selection == "BGE":
        embedding_model_type = "huggingface_api"
        model_name = config.get("embedding_model_bge", "BAAI/bge-large-en-v1.5")
    elif model_selection == "E5-large":
        embedding_model_type = "custom"
        model_name = config.get("embedding_model_e5large", "intfloat/e5-large-v2")
    else:  # Default to all-MiniLM
        embedding_model_type = "sentence_transformer"
        model_name = "all-MiniLM-L6-v2"
    
    # Create the pipeline with the correct parameters
    pipeline = CVProcessingPipeline(
        config=config,
        embedding_model_type=embedding_model_type,
        use_classifier=use_classifier
    )
    
    # Manually update the embedding model
    if model_selection == "BGE":
        pipeline.embedding_model = create_embedding_model(
            model_type="huggingface_api",
            model_name=model_name
        )
    elif model_selection == "E5-large":
        pipeline.embedding_model = create_embedding_model(
            model_type="custom",
            model_name=model_name
        )
    else:
        pipeline.embedding_model = create_embedding_model(
            model_type="sentence_transformer",
            model_name=model_name
        )
    
    # Reinitialize vector store with new embedding model
    persist_dir = config.get("vector_store_directory", "output/vector_store")
    vector_store_type = config.get("vector_store_type", "chroma")
    pipeline.vector_store = create_vector_store(
        embedding_model=pipeline.embedding_model,
        store_type=vector_store_type,
        collection_name="cv_sections",
        persist_directory=persist_dir
    )
    
    return pipeline

# Header
st.title("CV-StructRAG: Resume Structure Extractor")
st.markdown("""
This tool extracts structured information from CV/Resume PDFs using an intelligent parsing system.
Upload a PDF to extract its contents into a structured JSON format.
""")

# Sidebar
st.sidebar.header("Settings")

# Add a button to clear the cache
if st.sidebar.button("🔄 Reload Models"):
    st.experimental_singleton.clear()
    st.experimental_memo.clear()
    st.info("Cache cleared! Reloading models...")
    st.rerun()

# New model selection dropdown
model_selection = st.sidebar.selectbox(
    "Embedding Model",
    ["all-MiniLM", "BGE", "E5-large"],
    index=0,
    help="Select the embedding model to use for processing"
)

use_classifier = st.sidebar.checkbox("Use Section Classifier", value=False)

# Add info about layout-aware processing being enabled by default
st.sidebar.info("✓ Layout-aware processing (LayoutLMv3) is enabled by default for better document structure understanding.")

# Create pipeline with selected model
pipeline = get_pipeline(
    model_selection=model_selection, 
    use_classifier=use_classifier, 
    _version=1
)

# Show which model is being used
with st.sidebar.expander("Model Details", expanded=False):
    if model_selection == "BGE":
        st.info("Using BGE Large model from BAAI")
        st.caption("A powerful bilingual embedding model optimized for retrieval")
    elif model_selection == "E5-large":
        st.info("Using E5-large model from intfloat")
        st.caption("High-quality text embeddings with strong performance on search tasks")
    else:
        st.info("Using all-MiniLM model from Sentence Transformers")
        st.caption("Lightweight model with good performance for general purpose embeddings")

# File uploader
uploaded_file = st.file_uploader("Upload a Resume/CV (PDF)", type="pdf")

# Process the PDF
if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    
    with st.spinner("Processing PDF..."):
        try:
            # Process the PDF
            result = pipeline.process_pdf(pdf_path, save_output=False)
            
            # Check if there was an error during processing
            if "metadata" in result and "error" in result["metadata"]:
                st.error(f"Error processing PDF: {result['metadata']['error']}")
                st.write("Showing partial results if available:")
            
            # Debug information
            if st.checkbox("Show debug info"):
                structured_data = result.get("structured_data", {})
                
                st.write("Data structure:")
                st.json({
                    "Keys in result": list(result.keys()),
                    "Keys in structured_data": list(structured_data.keys()) if structured_data else []
                })
                
                # Check projects structure
                if "projects" in structured_data:
                    st.write(f"Projects count: {len(structured_data['projects'])}")
                    for i, project in enumerate(structured_data["projects"]):
                        st.write(f"Project {i} keys: {list(project.keys())}")
                        st.write(f"Project {i} description type: {type(project.get('description', []))}")
            
            # Main layout with tabs
            tab1, tab2, tab3 = st.tabs(["Structured Output", "Sections", "Raw JSON"])
            
            # Tab 1: Structured Output
            with tab1:
                st.subheader("Structured Data")
                
                # Create expandable sections for each category
                structured_data = result["structured_data"]
                
                # Display contact information first if available
                if "Contact" in structured_data:
                    with st.expander("Contact Information", expanded=True):
                        contact_info = structured_data["Contact"]
                        for key, value in contact_info.items():
                            st.text(f"{key.capitalize()}: {value}")
                
                # Display other sections
                for section_name, section_data in structured_data.items():
                    if section_name == "Contact":
                        continue  # Already displayed
                    
                    with st.expander(section_name, expanded=True):
                        # Different display based on section type
                        if section_name == "skills" and isinstance(section_data, list):
                            # Skills list
                            for item in section_data:
                                st.markdown(f"• {item}")
                        elif section_name == "projects" and isinstance(section_data, list):
                            # Projects list
                            for project in section_data:
                                st.markdown(f"**{project.get('title', '')}**")
                                # Handle project description which could be a list or string
                                if isinstance(project.get('description', []), list):
                                    for item in project['description']:
                                        if item:  # Skip empty items
                                            st.markdown(f"• {item}")
                                else:
                                    st.text(project.get('description', ''))
                                st.markdown("---")
                        elif isinstance(section_data, list):
                            # Simple list (like Skills)
                            for item in section_data:
                                if isinstance(item, str):
                                    st.markdown(f"• {item}")
                                elif isinstance(item, dict):
                                    # Handle dictionary items
                                    for key, value in item.items():
                                        st.markdown(f"**{key}**: {value}")
                                else:
                                    st.markdown(f"• {str(item)}")
                        elif isinstance(section_data, dict):
                            # Nested structure (like Experience)
                            for subsection, items in section_data.items():
                                st.markdown(f"**{subsection}**")
                                if isinstance(items, list):
                                    for item in items:
                                        st.markdown(f"• {item}")
                                else:
                                    st.text(items)
                                st.markdown("---")
                        else:
                            # Plain text or other format
                            st.text(str(section_data))
            
            # Tab 2: Sections View
            with tab2:
                st.subheader("Extracted Sections")
                
                # Show hierarchical chunks
                for i, section in enumerate(result["hierarchical_chunks"]):
                    with st.expander(f"{section['heading']}", expanded=False):
                        st.markdown(f"**Section Content:**")
                        st.text(section["content"])
                        
                        if section.get("subsections"):
                            st.markdown("**Subsections:**")
                            for subsection in section["subsections"]:
                                st.markdown(f"- **{subsection['heading']}**")
                                st.text(subsection["content"])
            
            # Tab 3: Raw JSON
            with tab3:
                st.subheader("Raw Structured JSON")
                st.json(result["structured_data"])
                
                # Download button for JSON
                json_str = json.dumps(result["structured_data"], indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{Path(uploaded_file.name).stem}_structured.json",
                    mime="application/json"
                )
        
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        
        finally:
            # Clean up temporary file
            os.unlink(pdf_path)

# Search functionality
st.markdown("---")
st.subheader("Search in Processed Documents")

col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input("Search Query", placeholder="Enter search query...")
with col2:
    section_filter = st.selectbox(
        "Filter by Section",
        ["All Sections", "Experience", "Education", "Skills", "Projects", "Summary", "Contact"],
        index=0
    )

if search_query:
    with st.spinner("Searching..."):
        # Convert "All Sections" to None for the filter
        filter_value = None if section_filter == "All Sections" else section_filter
        
        # Perform search
        try:
            search_results = pipeline.search_sections(search_query, filter_value, k=5)
            
            if search_results:
                for i, result in enumerate(search_results):
                    score = result["score"]
                    text = result["text"]
                    metadata = result["metadata"]
                    
                    with st.expander(f"Result {i+1} - Score: {score:.2f} - {metadata.get('heading', '')}"):
                        st.markdown(f"**Section:** {metadata.get('normalized_heading', 'Unknown')}")
                        if "parent_section" in metadata:
                            st.markdown(f"**Parent Section:** {metadata['parent_section']}")
                        st.markdown("**Content:**")
                        st.text(text)
            else:
                st.info("No results found.")
        except Exception as e:
            st.error(f"Error during search: {str(e)}")

# Footer
st.markdown("---")
st.markdown("CV-StructRAG - Structured Information Retrieval from CVs")
