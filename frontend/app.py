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

# Page configuration
st.set_page_config(
    page_title="CV-StructRAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    config = load_config()
    return CVProcessingPipeline(config=config)

pipeline = get_pipeline()

# Header
st.title("CV-StructRAG: Resume Structure Extractor")
st.markdown("""
This tool extracts structured information from CV/Resume PDFs using an intelligent parsing system.
Upload a PDF to extract its contents into a structured JSON format.
""")

# Sidebar
st.sidebar.header("Settings")
embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["sentence_transformer", "custom", "openai"],
    index=0
)

model_name = st.sidebar.text_input(
    "Model Name", 
    value="all-MiniLM-L6-v2"
)

use_classifier = st.sidebar.checkbox("Use Section Classifier", value=False)

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
                        if isinstance(section_data, list):
                            # Simple list (like Skills)
                            for item in section_data:
                                st.markdown(f"• {item}")
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
