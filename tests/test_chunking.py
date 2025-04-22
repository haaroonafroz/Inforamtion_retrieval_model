import os
import pytest
from src.chunking import ResumeChunker

# Mock PDF text for testing
MOCK_PDF_TEXT = """
JOHN DOE
email@example.com | (123) 456-7890 | linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years developing web applications.

EXPERIENCE
Senior Software Engineer - ABC Tech
January 2020 - Present
• Led development of RESTful APIs using Python Flask
• Optimized database queries improving performance by 30%

Software Developer - XYZ Solutions
March 2017 - December 2019
• Developed frontend components using React.js
• Implemented CI/CD pipelines with Jenkins

EDUCATION
Master of Computer Science - University of Technology
2015 - 2017
• GPA: 3.8/4.0

Bachelor of Science in Computer Engineering - State University
2011 - 2015

SKILLS
Python, JavaScript, React, Flask, SQL, Git, Docker, Kubernetes
"""

class TestResumeChunker:
    """Test case for ResumeChunker class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.chunker = ResumeChunker()
    
    def test_is_section_heading(self):
        """Test section heading detection"""
        # Test standard section headings
        assert self.chunker.is_section_heading("EXPERIENCE") == True
        assert self.chunker.is_section_heading("Education") == True
        assert self.chunker.is_section_heading("Skills") == True
        
        # Test non-section text
        assert self.chunker.is_section_heading("John Doe") == False
        assert self.chunker.is_section_heading("email@example.com") == False
        assert self.chunker.is_section_heading("• Led development of RESTful APIs") == False
    
    def test_is_job_title(self):
        """Test job title detection"""
        # Test valid job titles
        is_job, info = self.chunker.is_job_title("Senior Software Engineer - ABC Tech")
        assert is_job == True
        assert info["title"] == "Senior Software Engineer"
        assert info["company"] == "ABC Tech"
        
        is_job, info = self.chunker.is_job_title("Software Developer at XYZ Solutions")
        assert is_job == True
        
        # Test invalid job titles
        is_job, _ = self.chunker.is_job_title("Python, JavaScript, React")
        assert is_job == False
    
    def test_extract_bullet_points(self):
        """Test bullet point extraction"""
        text = """
        • First bullet point
        • Second bullet point
        - Third bullet point
        """
        bullet_points = self.chunker.extract_bullet_points(text)
        assert len(bullet_points) == 3
        assert "First bullet point" in bullet_points
        assert "Second bullet point" in bullet_points
        assert "Third bullet point" in bullet_points
    
    def test_chunk_resume(self):
        """Test resume chunking with plain text"""
        chunks = self.chunker.chunk_resume(MOCK_PDF_TEXT)
        
        # Check if main sections are detected
        section_headings = [section["heading"] for section in chunks]
        assert "PROFESSIONAL SUMMARY" in section_headings
        assert "EXPERIENCE" in section_headings
        assert "EDUCATION" in section_headings
        assert "SKILLS" in section_headings
        
        # Check subsections in Experience section
        experience_section = next((s for s in chunks if s["heading"] == "EXPERIENCE"), None)
        assert experience_section is not None
        assert len(experience_section["subsections"]) == 2
        
        # Check if job titles are extracted correctly
        job_titles = [sub["heading"] for sub in experience_section["subsections"]]
        assert "Senior Software Engineer - ABC Tech" in job_titles
        assert "Software Developer - XYZ Solutions" in job_titles
    
    def test_normalize_section_heading(self):
        """Test section heading normalization"""
        assert self.chunker.normalize_section_heading("EXPERIENCE") == "Experience"
        assert self.chunker.normalize_section_heading("Work Experience") == "Experience"
        assert self.chunker.normalize_section_heading("Professional Experience") == "Experience"
        assert self.chunker.normalize_section_heading("EDUCATION") == "Education"
        assert self.chunker.normalize_section_heading("SKILLS") == "Skills"
        assert self.chunker.normalize_section_heading("Technical Skills") == "Skills"
        assert self.chunker.normalize_section_heading("UNKNOWN SECTION") == "Other"
    
    def test_convert_to_output_format(self):
        """Test conversion to output JSON format"""
        # Create sample hierarchical chunks
        chunks = [
            {
                "type": "section",
                "heading": "EXPERIENCE",
                "content": "",
                "metadata": {"normalized_heading": "Experience"},
                "subsections": [
                    {
                        "type": "subsection",
                        "heading": "Software Engineer - ABC Corp",
                        "title": "Software Engineer",
                        "company": "ABC Corp",
                        "content": "• Built APIs\n• Developed UI",
                        "bullet_points": ["Built APIs", "Developed UI"]
                    }
                ]
            },
            {
                "type": "section",
                "heading": "SKILLS",
                "content": "Python, JavaScript, React",
                "metadata": {"normalized_heading": "Skills"},
                "subsections": []
            }
        ]
        
        output = self.chunker.convert_to_output_format(chunks)
        
        # Check structure
        assert "Experience" in output
        assert "Skills" in output
        
        # Check experience section
        assert "Software Engineer - ABC Corp" in output["Experience"]
        assert len(output["Experience"]["Software Engineer - ABC Corp"]) == 2
        assert "Built APIs" in output["Experience"]["Software Engineer - ABC Corp"]
        
        # Check skills section
        assert isinstance(output["Skills"], list)
        assert "Python" in output["Skills"] or "Python, JavaScript, React" in output["Skills"] 