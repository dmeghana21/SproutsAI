#  Sprouts - AI-Powered Candidate Recommendation Engine

A sophisticated AI-powered system that intelligently matches job descriptions with candidate resumes using advanced natural language processing and machine learning techniques.

##  Features

### Core Capabilities
- **Intelligent Keyword Extraction**: Uses GPT-4 to extract relevant technical keywords from job descriptions without hardcoding
- **Semantic Similarity Matching**: Leverages state-of-the-art embedding models for accurate candidate-job matching
- **Comprehensive Keyword Matching**: Searches through complete resume text with intelligent variations and abbreviations
- **AI-Generated Summaries**: Provides personalized fit summaries for each candidate (GPT-4o)
- **Multi-Format Support**: Handles PDF resume format
- **Real-time Processing**: Fast, efficient processing with detailed logging
- **Adaptive Top-N Results**: Returns top 5–10 candidates based on how many resumes are uploaded
- **Smart Name Extraction**: Extracts names from resume content first (top lines), with a clean filename fallback

### Technical Highlights
- **Adaptive Keyword Extraction**: Dynamic keyword extraction using GPT-4's contextual understanding
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
pip install -r Sprouts/requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Application
```bash
streamlit run Sprouts/app.py
```

##  Architecture

### Project Structure
```
Sprouts/
├── README.md                     # Project documentation (this file)
├── logs/                         # Processing logs (created at runtime)
│   └── agent_thoughtlog.txt
├── outputs/                      # Generated recommendations (created at runtime)
│   └── recommended_candidates.txt
└── Sprouts/
    ├── app.py                    # Main Streamlit application
    ├── requirements.txt          # Python dependencies
    ├── test_openai.py            # OpenAI API testing
    └── utils/
        ├── agent.py              # Core recommendation engine
        ├── parser.py             # PDF file parsing
        ├── embedding.py          # Text embedding generation
        ├── similarity.py         # Cosine similarity computation
        └── gpt_summary.py        # AI-generated candidate summaries
```

### Core Components

#### 1. **Agent Engine** (`utils/agent.py`)
- **`extract_candidate_name()`**: Smart name extraction and formatting from filenames
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
- Supports PDF format
- Robust error handling for corrupted files

#### 3. **Embedding System** (`utils/embedding.py`)
- Uses SentenceTransformer 'all-MiniLM-L6-v2' model (default)
- Model caching for performance optimization
- Support for multiple embedding providers (OpenAI, BAAI) - available but not used in main flow

#### 4. **Similarity Computation** (`utils/similarity.py`)
- Cosine similarity calculation
- Efficient vector operations
- Normalized similarity scores

#### 5. **AI Summaries** (`utils/gpt_summary.py`)
- GPT-4o powered candidate fit summaries
- Personalized explanations for each match
- Error handling for API failures

##  Usage

### 1. Launch the Application
```bash
streamlit run Sprouts/app.py
```

### 2. Input Job Description
- Paste or type the job description in the text area
- The system will automatically extract relevant technical keywords using GPT-4

### 3. Upload Resumes
- Upload multiple resume files
- Supported formats: `.pdf`
- Maximum file size: 200MB per file (Streamlit default unless configured)
- **Smart Name Extraction**: Candidate names are automatically extracted and formatted from filenames

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

 **[Live Candidate Recommendation Engine App URL](https://dmeghana21-sprouts-ai--sproutsapp-urldgi.streamlit.app)**

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

### 2. **Smart Name Extraction**
```python
# Prefer name from resume text; fallback to filename
name = extract_candidate_name_from_text(resume_text) or extract_candidate_name(filename)
# Examples:
# 'piotr_migdal_resume.pdf' → 'Piotr Migdal'
# 'cristian_garcia.pdf' → 'Cristian Garcia'
# 'john-doe-cv.pdf' → 'John Doe'
```

### 3. **Comprehensive Keyword Matching**
```python
# Search through complete resume text with intelligent variations
matched_keywords = find_keyword_matches(resume_text, job_keywords)
# Handles variations: 'python' → ['py', 'python3'], 'javascript' → ['js', 'node.js']
# Returns: ['python', 'openai', 'fastapi'] - keywords found in resume
```

### 4. **Semantic Embedding**
```python
# Generate embeddings for job and resume texts
job_embedding = embed_text(job_text, model)
resume_embedding = embed_text(resume_text, model)
```

### 5. **Similarity Calculation**
```python
# Compute cosine similarity between job and resume
similarity = calculate_cosine_similarity(job_embedding, resume_embedding)
```

### 6. **AI Summary Generation**
```python
# Generate personalized fit summary (GPT-4o)
summary = generate_fit_summary(job_description, resume_text)
```

### 7. **Structured Logging**
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

### Candidate Results Display
```
Top N Recommended Candidates

1. Aleksandr Nikitin
AI Summary: Aleksandr Nikitin is a strong fit for the role at Pragmatike due to his extensive experience in developing and deploying data-driven solutions, as demonstrated by his work at Tochka Bank where he led the design and implementation of machine learning models. His proficiency in Python and familiarity with cloud environments, combined with his strong communication and team collaboration skills, align well with the job's requirements. Additionally, his background in leading projects in high-paced environments and his commitment to professional development in AI/ML systems make him an excellent candidate for contributing to Pragmatike's innovative projects in Cloud Computing, Blockchain, and Artificial Intelligence.

Keyword Matches (5): ai, machine learning, ml, python, go

Score:
- Cosine Similarity: 0.5751

---

2. Serkov Vladislav
AI Summary: Vladislav Serkov is a highly experienced Senior AI/ML Engineer who is an excellent fit for the role at Pragmatike, given his extensive background in deploying production-grade machine learning systems and optimizing model performance. He has a proven track record in MLOps workflows and has worked in fast-paced, innovation-driven environments, aligning well with the company's need for expertise in applied AI and cutting-edge technologies. His proficiency in Python, PyTorch, and cloud environments, combined with his strong communication and collaboration skills, make him an ideal candidate to contribute to Pragmatike's projects in AI, Cloud Computing, and Blockchain.

Keyword Matches (9): ai, machine learning, ml, python, pytorch, transformers, aws, docker, ci/cd

Score:
- Cosine Similarity: 0.5234
```

### Candidate Data Structure
```json
{
  "name": "Aleksandr Nikitin",
  "score": 0.5751,
  "similarity": 0.5751,
  "keyword_matches": ["ai", "machine learning", "ml", "python", "go"],
  "keyword_count": 5,
  "summary": "Aleksandr Nikitin is a strong fit for the role..."
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
- **GPT Models**:
  - GPT-4 for keyword extraction
  - GPT-4o for candidate fit summaries
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

- **OpenAI**: For GPT-4 and GPT-4o APIs
- **SentenceTransformers**: For semantic similarity computation
- **Streamlit**: For the web application framework
- **PyMuPDF**: For PDF processing capabilities
- **Hugging Face**: For transformer models and utilities

---

**Built for intelligent candidate matching**
