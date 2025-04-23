# Codebase Refactoring

This document outlines the refactoring changes made to improve the codebase organization and maintainability.

## Key Changes

### 1. Merged Layout-Aware Chunking

- **Layout-aware capabilities** have been integrated directly into the core `ResumeChunker` class in `src/chunking.py`
- The separate `LayoutAwareResumeChunker` class has been removed
- Layout-aware processing is now the default approach, providing better document structure recognition

### 2. Centralized Section Mappings

- Section heading mappings are now centralized in `config.py`
- The `get_section_name()` function from `config.py` is used across the codebase for consistent heading normalization
- This eliminates duplication and ensures consistent classification of resume sections

### 3. Reorganized Utility Functions

- Utility functions have been organized for better reusability
- Text extraction functions like `extract_text_info_from_pdf` are properly located in `chunking.py` where they're used
- General PDF processing functions remain in `utils.py`

### 4. Simplified Pipeline 

- `CVProcessingPipeline` class has been simplified to use the enhanced `ResumeChunker`
- The option to toggle between standard and layout-aware chunking has been removed
- Layout-aware processing is now enabled by default for all processing

## Benefits

1. **Reduced Code Duplication**: Similar functionality is now defined in one place
2. **Improved Maintainability**: Fewer files and centralized configuration
3. **Enhanced Performance**: Layout-aware processing as the default provides better structure detection
4. **Cleaner API**: Simplified interface with fewer options leads to more predictable behavior

## Implementation Details

- The `ResumeChunker` now incorporates both traditional heading-based and layout-based parsing
- The chunker automatically falls back to traditional parsing if layout model initialization fails
- Section normalization uses centralized mappings from `config.py`
- Pipeline processing logic has been streamlined for better error handling and cleaner code 