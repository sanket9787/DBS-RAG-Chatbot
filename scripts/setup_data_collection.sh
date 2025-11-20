#!/bin/bash
# DBS Data Collection Setup Script
# Phase 2: Data Collection - Environment Setup

echo "🚀 Setting up DBS Data Collection Environment..."

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/raw/pdfs
mkdir -p data/processed
mkdir -p logs

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r scripts/requirements.txt

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
print('NLTK data downloaded successfully')
"

# Create sample PDF directory with instructions
echo "📄 Creating PDF directory with instructions..."
cat > data/raw/pdfs/README.md << 'EOF'
# PDF Documents Directory

Place DBS PDF documents in this directory for processing.

## Supported File Types
- .pdf - PDF documents
- .doc - Microsoft Word documents
- .docx - Microsoft Word documents (newer format)

## Recommended Documents
- Course prospectuses
- Student handbooks
- Application forms
- Policy documents
- Academic calendars

## File Naming
Use descriptive names like:
- undergraduate_prospectus_2024.pdf
- student_handbook_2024.pdf
- admissions_guide_2024.pdf

## Processing
PDFs will be automatically processed when you run the data collection pipeline.
EOF

# Create environment configuration
echo "⚙️ Creating environment configuration..."
cat > .env << 'EOF'
# DBS Data Collection Environment Variables
# Copy this file to .env and fill in your actual values

# OpenAI Configuration (Required for embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION_NAME=dbs_documents

# Web Scraping Configuration
DBS_WEBSITE_URL=https://www.dbs.ie
SCRAPING_DELAY=1.0
MAX_CONCURRENT_REQUESTS=5

# Data Processing Configuration
MIN_QUALITY_SCORE=0.3
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Logging Configuration
LOG_LEVEL=INFO
EOF

# Create data collection configuration
echo "📋 Creating data collection configuration..."
cat > data_collection_config.json << 'EOF'
{
  "web_scraping": {
    "enabled": true,
    "base_url": "https://www.dbs.ie",
    "delay": 1.0,
    "max_pages": 100
  },
  "pdf_processing": {
    "enabled": true,
    "input_dir": "data/raw/pdfs",
    "max_file_size": 10485760
  },
  "data_cleaning": {
    "enabled": true,
    "min_quality_score": 0.3,
    "remove_duplicates": true
  },
  "quality_assurance": {
    "enabled": true,
    "min_quality_score": 0.5,
    "validate_urls": true
  },
  "knowledge_base": {
    "enabled": true,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "collection_name": "dbs_documents"
  }
}
EOF

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/*.py

echo "✅ Data Collection Environment Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Add your OpenAI API key to .env file"
echo "2. Add PDF documents to data/raw/pdfs/ directory"
echo "3. Run the data collection pipeline:"
echo "   python scripts/run_data_collection.py"
echo ""
echo "📁 Directory Structure Created:"
echo "   data/raw/pdfs/          - Place PDF documents here"
echo "   data/processed/         - Processed data will be saved here"
echo "   logs/                   - Log files will be saved here"
echo ""
echo "🔧 Configuration Files:"
echo "   .env                    - Environment variables"
echo "   data_collection_config.json - Pipeline configuration"
echo ""
echo "🚀 Ready to start data collection!"
