#  Sprouts - AI-Powered Candidate Recommendation Engine

A sophisticated AI-powered system that intelligently matches job descriptions with candidate resumes using advanced natural language processing and machine learning techniques.

##  Features

### Core Capabilities
- **Intelligent Keyword Extraction**: Uses GPT-4 to extract relevant technical keywords from job descriptions without hardcoding
- **Semantic Similarity Matching**: Leverages state-of-the-art embedding models for accurate candidate-job matching
- **Comprehensive Keyword Matching**: Searches through complete resume text with intelligent variations and abbreviations
- **AI-Generated Summaries**: Provides personalized fit summaries for each candidate
- **Multi-Format Support**: Handles both PDF and TXT resume formats
- **Real-time Processing**: Fast, efficient processing with detailed logging
- **Robust Fallback System**: Graceful degradation when GPT-4 is unavailable

### Technical Highlights
- **No Hardcoding**: Dynamic keyword extraction using GPT-4's contextual understanding
- **Advanced NLP**: Uses SentenceTransformer embeddings for semantic similarity
- **Comprehensive Logging**: Detailed audit trail with decorative separators
- **Modular Architecture**: Clean, maintainable code structure
- **Scalable Design**: Easy to extend and customize
- **Intelligent Variations**: Handles common abbreviations and synonyms

##  Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (for embedding models)
- Internet connection (for OpenAI API)

### Python Dependencies
```
streamlit>=1.28.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
PyMuPDF>=1.23.0
openai>=1.0.0
python-dotenv>=1.0.0
torch>=2.0.0
transformers>=4.30.0
```

##  Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Sprouts
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Application
```bash
streamlit run app.py
```

##  Architecture

### Project Structure
```
Sprouts/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── test_openai.py        # OpenAI API testing
├── utils/
│   ├── agent.py          # Core recommendation engine
│   ├── parser.py         # PDF/TXT file parsing
│   ├── embedding.py      # Text embedding generation
│   ├── similarity.py     # Cosine similarity computation
│   └── gpt_summary.py    # AI-generated candidate summaries
├── logs/                 # Processing logs
└── outputs/              # Generated recommendations
```

### Core Components

#### 1. **Agent Engine** (`utils/agent.py`)
- **`extract_technical_keywords()`**: GPT-4 powered keyword extraction with fallback
- **`extract_keywords_enhanced()`**: Rule-based fallback keyword extraction
- **`find_keyword_matches()`**: Comprehensive keyword matching with variations
- **`embed_text()`**: Generates text embeddings
- **`calculate_cosine_similarity()`**: Computes similarity scores
- **`run_agentic_flow()`**: Main orchestration function
- **`log_agentic_flow_start()`**: Structured logging for flow initiation
- **`log_step_header()`**: Step headers with decorative separators
- **`log_technical_keywords()`**: Clean keyword extraction logging
- **`log_resume_processing()`**: Resume processing summary logging
- **`log_similarity_scores()`**: Aligned similarity scores display
- **`log_agentic_flow_end()`**: Flow completion with top candidate highlight
- **`log_agent_thought()`**: Detailed logging with timestamps
- **`log_step_completion()`**: Decorative separator logging

#### 2. **File Parser** (`utils/parser.py`)
- Supports PDF and TXT file formats
- Robust error handling for corrupted files
- UTF-8 encoding support

#### 3. **Embedding System** (`utils/embedding.py`)
- Uses SentenceTransformer 'all-MiniLM-L6-v2' model
- Model caching for performance optimization
- Support for multiple embedding providers (OpenAI, BAAI)

#### 4. **Similarity Computation** (`utils/similarity.py`)
- Cosine similarity calculation
- Efficient vector operations
- Normalized similarity scores

#### 5. **AI Summaries** (`utils/gpt_summary.py`)
- GPT-4 powered candidate fit summaries
- Personalized explanations for each match
- Error handling for API failures

##  Usage

### 1. Launch the Application
```bash
streamlit run app.py
```

### 2. Input Job Description
- Paste or type the job description in the text area
- The system will automatically extract relevant technical keywords using GPT-4

### 3. Upload Resumes
- Upload multiple PDF or TXT resume files
- Supported formats: `.pdf`, `.txt`
- Maximum file size: 10MB per file

### 4. Generate Recommendations
- Click "Recommend Candidates" to start processing
- View real-time processing logs with decorative separators
- Get detailed results with keyword matches

### 5. Review Results
- **Cosine Similarity Score**: Semantic match quality (0-1)
- **Keyword Matches**: Specific skills/technologies found in each resume
- **AI Summary**: Personalized fit explanation
- **Download Results**: Export recommendations and logs
  
##  Deployment

The app is publicly deployed and accessible here:

 **[Live App URL](https://dmeghana21-sprouts-ai--sproutsapp-urldgi.streamlit.app)**

###  API Key Requirement

- The OpenAI API key is already securely stored in the deployed app via **Streamlit secrets**.
- **No setup is required for end users.**
- If you're running the app locally, you'll need to create a `.env` or use `secrets.toml` with your own key.

##  How It Works

### 1. **Intelligent Keyword Extraction**
```python
# GPT-4 analyzes job description and extracts relevant technical terms
keywords = extract_technical_keywords(job_description, openai_api_key)
# Returns: ['python', 'openai', 'langchain', 'fastapi', 'devops', ...]

# Fallback to rule-based extraction if GPT-4 fails
if gpt_fails:
    keywords = extract_keywords_enhanced(job_description)
```

### 2. **Comprehensive Keyword Matching**
```python
# Search through complete resume text with intelligent variations
matched_keywords = find_keyword_matches(resume_text, job_keywords)
# Handles variations: 'python' → ['py', 'python3'], 'javascript' → ['js', 'node.js']
# Returns: ['python', 'openai', 'fastapi'] - keywords found in resume
```

### 3. **Semantic Embedding**
```python
# Generate embeddings for job and resume texts
job_embedding = embed_text(job_text, model)
resume_embedding = embed_text(resume_text, model)
```

### 4. **Similarity Calculation**
```python
# Compute cosine similarity between job and resume
similarity = calculate_cosine_similarity(job_embedding, resume_embedding)
```

### 5. **AI Summary Generation**
```python
# Generate personalized fit summary
summary = generate_fit_summary(job_description, resume_text)
```

### 6. **Structured Logging**
```python
# Comprehensive audit trail with grouped sections
log_agentic_flow_start()           # START section
log_step_header()                  # Step headers with separators
log_technical_keywords()           # Clean keyword logging
log_resume_processing()            # Resume count summary
log_similarity_scores()            # Aligned similarity display
log_agentic_flow_end()             # END with top candidate highlight
```

##  Output Format

### Candidate Results
```json
{
  "name": "John Doe",
  "score": 0.8234,
  "similarity": 0.8234,
  "keyword_matches": ["python", "openai", "langchain", "fastapi"],
  "keyword_count": 4,
  "summary": "Strong technical match with relevant AI/ML experience..."
}
```

### Log Files
- **Agent Thought Log**: Detailed processing steps with decorative separators
- **Recommendations File**: Formatted candidate rankings and details

##  Configuration

### Environment Variables
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Model Configuration
- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformer)
- **GPT Model**: `gpt-4` (OpenAI)
- **Temperature**: 0.1 (for consistent keyword extraction)
- **Fallback Model**: Rule-based keyword extraction

##  Performance

### Processing Speed
- **Small batch (1-5 resumes)**: ~10-30 seconds
- **Medium batch (5-20 resumes)**: ~1-3 minutes
- **Large batch (20+ resumes)**: ~3-5 minutes

### Accuracy Metrics
- **Semantic Similarity**: High accuracy for technical role matching
- **Keyword Extraction**: Context-aware, no false positives
- **Keyword Matching**: Intelligent variations and abbreviations
- **Summary Quality**: Personalized, relevant fit explanations

##  Error Handling

### Robust Error Management
- **API Failures**: Graceful fallback to rule-based keyword extraction
- **File Processing**: Handles corrupted or unsupported file formats
- **Memory Management**: Efficient handling of large resume files
- **Network Issues**: Retry logic for API calls
- **Model Loading**: Caching for performance optimization

## Extensibility

### Easy Customization
- **New File Formats**: Add parsers in `utils/parser.py`
- **Different Embedding Models**: Configure in `utils/embedding.py`
- **Custom Scoring**: Modify similarity calculation in `utils/similarity.py`
- **Additional Features**: Extend `utils/agent.py` with new capabilities
- **Keyword Variations**: Add new variations in `find_keyword_matches()`

## Logging

### Comprehensive Audit Trail with Structured Format
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2025-08-07T18:39:46] START: Agentic flow initiated with technical keyword extraction and cosine similarity.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2025-08-07T18:39:46] EXTRACT_KEYWORDS: Extracting technical keywords from job description.
[2025-08-07T18:39:54] TECHNICAL_KEYWORDS: Extracted keywords: ['python', 'openai', 'langchain', 'fastapi', 'devops']
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2025-08-07T18:39:55] EMBEDDING_AND_SIMILARITY: Generating embeddings and computing cosine similarity scores.
[2025-08-07T18:39:55] RESUME_COUNT: Processed 3 resumes: ['John Doe', 'Jane Smith', 'Bob Johnson']
[2025-08-07T18:39:58] SIMILARITY_SCORES:
  1. John Doe: 0.8234
  2. Jane Smith: 0.7456
  3. Bob Johnson: 0.6123
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2025-08-07T18:39:58] END: Agentic flow completed. **Top candidate: John Doe with score 0.8234**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Logging Features
- **Structured Sections**: Grouped logs under clear headers (START, STEP 1, STEP 2, STEP 3, END)
- **Decorative Separators**: Visual section dividers for easy scanning
- **Aligned Similarity Scores**: Clean, descending order display of candidate scores
- **Top Candidate Highlight**: Bold formatting for the best match
- **Timestamped Entries**: ISO format timestamps for all operations
- **Clean Format**: No repetitive logs, focused on key information

##  Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings for all functions
- Include error handling
- Write clear commit messages

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support

### Common Issues
1. **OpenAI API Key**: Ensure your API key is valid and has sufficient credits
2. **File Upload**: Check file format and size limits
3. **Memory Issues**: Close other applications if processing large batches
4. **Network Problems**: Check internet connection for API calls
5. **Fallback Mode**: System will use rule-based extraction if GPT-4 fails

### Getting Help
- Check the logs in `logs/agent_thoughtlog.txt` for detailed error information
- Review the output files in `outputs/` for processing results
- Ensure all dependencies are properly installed
- Verify OpenAI API key is correctly set in environment variables

##  Acknowledgments

- **OpenAI**: For GPT-4 API and embedding capabilities
- **SentenceTransformers**: For semantic similarity computation
- **Streamlit**: For the web application framework
- **PyMuPDF**: For PDF processing capabilities
- **Hugging Face**: For transformer models and utilities

---

**Built with  for intelligent candidate matching**
