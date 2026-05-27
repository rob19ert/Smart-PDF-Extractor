# Smart PDF Extractor (Metallurgy & Engineering)

A specialized AI-powered tool for extracting, clustering, and classifying structured data from technical PDF documents, with a focus on metallurgical and engineering formulas.

## Project Overview

This project provides an automated pipeline to process PDF files, identify their structural elements (text, tables, formulas), and classify formulas into domain-specific categories (`physics`, `chemistry`, `math`). It is designed to handle complex metallurgical documents where symbols are often ambiguous (e.g., "V" for Vanadium vs. Volume) and equations can span multiple lines.

### Main Technologies
- **Python**: Core programming language.
- **unstructured[pdf]**: High-fidelity PDF partitioning using the `hi_res` strategy to extract elements and metadata.
- **scikit-learn**: Machine learning pipeline for formula classification (Logistic Regression + TF-IDF with character-level and word-level n-grams).
- **pdfplumber**: Used for extracting the base image of PDF pages and supporting coordinate mapping for visualization.
- **pandas**: Used to structure the extracted data and export it into Excel format.
- **matplotlib**: Visual verification with bounding box overlays.
- **joblib**: Serialization and deserialization of the trained scikit-learn models.

## Key Features

### 1. Hybrid Formula Classification
The system uses a sophisticated three-tier classification logic:
- **ML Layer**: A `LogisticRegression` model trained on a curated dataset of mathematical, physical, and chemical expressions and context keywords.
- **Structural Tie-Breaker**: A deterministic engine that analyzes the presence of chemical elements (via a comprehensive Periodic Table lookup), checks for specialized Greek symbols ($\sigma, \tau, \varepsilon$, etc.), and mathematical LaTeX commands (`\sum`, `\int`) to adjust classification weights dynamically.
- **Contextual Analysis**: Gathers surrounding text (2 blocks before, up to 5 blocks after) to find domain-specific keywords (e.g., "аустенит", "деформация", "скорость", "легирование") to assist in ambiguous symbol resolution.

### 2. Intelligent Structural Graph Clustering
To combat fragmentation often found in PDF extraction:
- Uses a **graph-based clustering algorithm** (Depth-First Search on a proximity matrix) to bind spatially close textual blocks together.
- Connects scattered words into coherent paragraphs based on vertical/horizontal gaps, indentation, font sizes, and structural boundaries.

### 3. Heuristic Formula Detection
The tool includes an `is_likely_formula` analyzer that uncovers formulas nested inside `NarrativeText` blocks based on:
- **Density Score**: Ratio of digits and technical symbols to alphabetic characters.
- **Pattern Matching**: Identification of LaTeX-style syntax, Cyrillic letter ratios, and specific equation markers (`=`, `→`, `≈`).
- **Table Scanning**: Analyzes tables to detect `FormulaTable` variations.

### 4. Visual & Data Output
- **Excel Export**: Structured breakdown of document elements including page numbers, types, full text, and text previews, saved to `*_blocks_final.xlsx`.
- **Visual Validation**: PNG overlays exported to the `output_images` directory with color-coded bounding boxes:
    - **Blue**: Text (Narrative, Title)
    - **Green shades**: Formulas (Light green for math, medium for physics)
    - **Red**: Tables
    - **Yellow**: Captions
- **GOST Standards**: Visualizations apply "Times New Roman" at 14pt, fulfilling engineering documentation standards.

## Building and Running

### Prerequisites
Install all necessary Python dependencies:
```bash
pip install -r requirements.txt
```
*Note: Due to the `unstructured` library's `hi_res` strategy, system libraries like `poppler-utils` and `tesseract-ocr` may be required on your system.*

### Training the Model
To train or update the AI classification model with new data from the global balanced dataset:
```bash
python train_metallurgy.py
```
This script processes the samples and creates/updates the `formula_classifier.pkl` model file.

### Running Extraction
To process all `.pdf` files present in the current directory and generate analytical outputs:
```bash
python main.py
```
The script will output Excel files in the working directory and visual validation images in the `output_images` directory.

## Development Conventions

- **Formatting Standards**: Visualizations strictly abide by the GOST standards (e.g., using Times New Roman font).
- **Naming Conventions**: Code uses English nomenclature, but handles Russian contexts heavily (bilingual labels, specific Cyrillic text checks).
- **Robust Model Loading**: The `SmartPDFExtractor` application runs gracefully without crashing even if the AI model (`formula_classifier.pkl`) is missing. It bypasses ML prediction and utilizes raw unstructured types instead.