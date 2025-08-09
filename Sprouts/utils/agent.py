import os
import re
from typing import Optional
from utils.parser import extract_text_from_file
from utils.embedding import get_embedding_model, generate_embedding, generate_chunk_embeddings
from utils.similarity import compute_similarity_scores
from utils.gpt_summary import generate_fit_summary
from datetime import datetime
import openai
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOG_PATH = os.path.join(PROJECT_ROOT, 'logs', 'agent_thoughtlog.txt')
RECOMMEND_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'recommended_candidates.txt')

DIVIDER = '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

ROLE_NOISE_WORDS = {
    'resume', 'cv', 'application', 'cover', 'letter', 'profile', 'final', 'updated', 'draft',
    'v1', 'v2', 'v3', 'copy', 'pdf', 'doc', 'docx', 'txt', 'new', 'old', 'latest', 'version',
    'fullstack', 'full-stack', 'full', 'stack', 'frontend', 'front-end', 'backend', 'back-end', 'developer',
    'engineer', 'software', 'swe', 'intern', 'senior', 'jr', 'sr', 'lead', 'manager', 'data',
    'scientist', 'analyst', 'ai', 'ml', 'dl', 'cloud', 'devops',
}

BRAND_WORDS = {
    'github', 'stackexchange', 'stackoverflow', 'linkedin', 'leetcode', 'kaggle',
    'medium', 'twitter', 'gmail', 'outlook', 'yahoo', 'google', 'facebook', 'instagram'
}

NAME_LINE_BLOCKLIST = {
    'resume', 'curriculum vitae', 'cover letter', 'portfolio', 'table of contents', 'experience',
    'education', 'skills', 'projects', 'summary', 'objective'
}

NAME_TOKEN_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z\-']+$")


def _clean_filename_to_tokens(filename: str) -> list:
    base = os.path.splitext(filename)[0]
    base = base.replace('.', ' ')
    base = re.sub(r'[\-_]+', ' ', base)
    base = re.sub(r'\s+', ' ', base).strip()
    tokens = [t for t in base.split(' ') if t]
    tokens_filtered = [t for t in tokens if t.lower() not in ROLE_NOISE_WORDS | BRAND_WORDS]
    tokens_filtered = [t for t in tokens_filtered if NAME_TOKEN_PATTERN.match(t)]
    return tokens_filtered


def _seems_name_token(token: str) -> bool:
    tl = token.lower()
    if len(tl) < 2:
        return False
    if token.isupper():
        return False
    if tl in ROLE_NOISE_WORDS or tl in BRAND_WORDS:
        return False
    if not any(v in tl for v in 'aeiou'):
        return False
    return True


def _name_from_filename(filename: str) -> str:
    tokens = _clean_filename_to_tokens(filename)
    name_tokens = [t for t in tokens if _seems_name_token(t)]
    if len(name_tokens) >= 2:
        return f"{name_tokens[0].title()} {name_tokens[1].title()}"
    if len(name_tokens) == 1:
        return name_tokens[0].title()
    # Fallback: best-effort title-cased filename without separators
    fallback = os.path.splitext(filename)[0]
    fallback = re.sub(r'[\-_]+', ' ', fallback)
    fallback = re.sub(r'\s+', ' ', fallback).strip()
    parts = [p for p in fallback.split(' ') if p and p.lower() not in ROLE_NOISE_WORDS | BRAND_WORDS]
    if len(parts) >= 2:
        return f"{parts[0].title()} {parts[1].title()}"
    if parts:
        return parts[0].title()
    return fallback.title()


def extract_candidate_name(filename: str, resume_text: Optional[str] = None) -> str:
    """
    Extract and format candidate name, prioritizing filename; fallback to resume text when needed.
    Returns: 'First Last' when possible, title-cased.
    """
    # First, try deriving from filename
    filename_guess = _name_from_filename(filename)
    if filename_guess and len(filename_guess.split()) >= 2:
        return filename_guess

    # Fallback to inferring from resume text
    if resume_text:
        inferred = infer_name_from_text(resume_text)
        if inferred:
            return inferred

    return filename_guess


def infer_name_from_text(resume_text: str) -> str:
    """
    Infer candidate name from resume text by scanning the first lines for a plausible name.
    """
    if not resume_text:
        return ""
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    # Look at the first ~20 non-empty lines
    for raw in lines[:20]:
        line = re.sub(r"[|,/\\]+", " ", raw)
        line = re.sub(r"\s+", " ", line).strip()
        lower = line.lower()
        if any(bl in lower for bl in NAME_LINE_BLOCKLIST):
            continue
        if any(b in lower for b in BRAND_WORDS):
            continue
        parts = [p for p in line.split(' ') if p]
        if 2 <= len(parts) <= 4 and all(NAME_TOKEN_PATTERN.match(p) for p in parts):
            # Avoid role and brand words
            if any(p.lower() in ROLE_NOISE_WORDS or p.lower() in BRAND_WORDS for p in parts):
                continue
            # Prefer Title Case tokens
            if parts[0][0].isalpha() and parts[1][0].isalpha():
                return f"{parts[0].title()} {parts[1].title()}"
    return ""


def log_agentic_flow_start():
    """Log the start of agentic flow with decorative separator."""
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"{DIVIDER}\n")
        f.write(f"[{datetime.now().isoformat()}] START: Agentic flow initiated with technical keyword extraction and cosine similarity.\n")

def log_step_header(step_name, description):
    """Log step header with decorative separator."""
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"{DIVIDER}\n")
        f.write(f"[{datetime.now().isoformat()}] {step_name}: {description}\n")

def log_technical_keywords(keywords):
    """Log extracted technical keywords."""
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] TECHNICAL_KEYWORDS: Extracted keywords: {keywords}\n")

def log_resume_processing(resume_names):
    """Log resume processing summary."""
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"{DIVIDER}\n")
        f.write(f"[{datetime.now().isoformat()}] RESUME_COUNT: Processed {len(resume_names)} resumes: {resume_names}\n")

def log_similarity_scores(candidates):
    """Log all similarity scores in sorted order."""
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] SIMILARITY_SCORES:\n")
        for i, candidate in enumerate(candidates, 1):
            score = candidate['similarity']
            name = candidate['name']
            f.write(f"  {i}. {name}: {score:.4f}\n")

def log_agentic_flow_end(top_candidate):
    """Log the end of agentic flow with top candidate highlight."""
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"{DIVIDER}\n")
        f.write(f"[{datetime.now().isoformat()}] END: Agentic flow completed. **Top candidate: {top_candidate['name']} with score {top_candidate['score']:.4f}**\n")
        f.write(f"{DIVIDER}\n")

def log_agent_thought(step, thought):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().isoformat()}] {step}: {thought}\n")

def log_step_completion(step_name):
    """Log step completion with decorative separator."""
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"{DIVIDER}\n")

def clear_logs():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(RECOMMEND_PATH), exist_ok=True)
    open(LOG_PATH, 'w').close()
    open(RECOMMEND_PATH, 'w').close()

def extract_keywords_enhanced(text, top_n=20):
    """Simplified fallback keyword extraction for when GPT-4 is unavailable."""
    # Basic technical terms for fallback
    TECHNICAL_TERMS = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'react', 'angular', 'vue',
        'node.js', 'django', 'flask', 'spring', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
        'docker', 'kubernetes', 'git', 'aws', 'azure', 'gcp', 'api', 'rest', 'graphql', 'microservices',
        'machine learning', 'deep learning', 'nlp', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
        'agile', 'scrum', 'ci/cd', 'jenkins', 'terraform', 'ansible', 'prometheus', 'grafana'
    }

    # Extract keywords by direct matching
    text_lower = text.lower()
    found_keywords = [term for term in TECHNICAL_TERMS if term in text_lower]

    return found_keywords

def extract_technical_keywords(job_text, openai_api_key):
    """
    Extract necessary technical keywords from job description using GPT.
    No hardcoding - GPT directly understands context and extracts relevant terms.

    Args:
        job_text (str): The job description text
        openai_api_key (str): OpenAI API key

    Returns:
        list: Clean list of relevant technical keywords for candidate shortlisting
    """
    if not openai_api_key:
        log_agent_thought('ERROR', 'OpenAI API key not provided for keyword extraction')
        return []

    client = openai.OpenAI(api_key=openai_api_key)

    prompt = (
        "You are an expert technical recruiter analyzing a job description. "
        "Extract ONLY the technical keywords, skills, tools, and technologies that are "
        "essential for candidate shortlisting and matching.\n\n"
        "Focus on:\n"
        "- Programming languages, frameworks, libraries\n"
        "- Tools, platforms, databases\n"
        "- Technical concepts, methodologies\n"
        "- Skills and responsibilities that indicate technical expertise\n\n"
        "IMPORTANT: Only extract terms that are explicitly mentioned or clearly implied "
        "in the job description. Do not guess or add generic terms.\n\n"
        "Return ONLY a comma-separated list of relevant technical keywords:\n\n"
        f"Job Description: {job_text}\n\n"
        "Technical Keywords:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )

        keywords_text = response.choices[0].message.content.strip()

        # Clean and process keywords
        keywords = []
        for kw in keywords_text.split(','):
            kw = kw.strip().lower()
            if kw and len(kw) > 1 and not kw.endswith('.'):
                keywords.append(kw)

        return keywords

    except Exception as e:
        log_agent_thought('ERROR', f'GPT keyword extraction failed: {e}')
        # Fallback to enhanced extraction
        return extract_keywords_enhanced(job_text)

def embed_text(text, model):
    """
    Embed text using the specified embedding model.

    Args:
        text (str): Text to embed
        model: Embedding model

    Returns:
        numpy.ndarray: Text embedding
    """
    return generate_embedding(text, model)

def calculate_cosine_similarity(job_embedding, resume_embedding):
    """
    Calculate cosine similarity between job and resume embeddings.

    Args:
        job_embedding: Job text embedding
        resume_embedding: Resume text embedding

    Returns:
        float: Cosine similarity score
    """
    similarities = compute_similarity_scores(job_embedding, [resume_embedding])
    return similarities[0]

def find_keyword_matches(resume_text, job_keywords):
    """
    Enhanced JD→Resume keyword matching with conservative normalization and vetted variants.
    - Case-insensitive
    - Multi-word phrases matched as phrases; single words use word boundaries
    - Handles a small set of safe variants (e.g., "data science"↔"data scientist", "jupyter notebook"↔"jupyter")
    - Strict terms must appear verbatim in the resume (except SAR alternative noted below)
    - No new keywords introduced; if uncertain, exclude

    Returns the JD keywords (original casing) that are supported by the resume text.
    """
    import re

    def normalize(text: str) -> str:
        t = text.lower()
        t = re.sub(r"[\-_]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    resume_lower = resume_text.lower()
    resume_norm = normalize(resume_text)

    matched_keywords = []

    # Terms that must appear exactly as written in the resume (case-insensitive)
    # Exception: allow 'sar' as an accepted alternative for 'synthetic aperture radar'
    strict_terms = {
        'synthetic aperture radar',
        'remote sensing',
        'intelligence community',
        'ts/sci',
        'ci poly',
        'databricks',
    }

    # Vetted equivalences for non-strict terms
    equivalence_map = {
        'data science': ['data science', 'data scientist'],
        'script development': ['script development', 'scripts', 'scripting'],
        'presentation': ['presentation', 'presented', 'presentations', 'lecturing', 'workshops'],
        'jupyter notebook': ['jupyter notebook', 'jupyter'],
        'machine learning': ['machine learning', 'ml'],
        'artificial intelligence': ['artificial intelligence', 'ai'],
        'api': ['api', 'apis', 'rest api', 'graphql'],
        'sql': ['sql', 'mysql', 'postgresql', 'sqlite', 'database'],
        'ci/cd': ['ci/cd', 'cicd', 'ci cd'],
        'javascript': ['javascript', 'js', 'node.js', 'nodejs'],
    }

    def contains_phrase(text: str, phrase: str) -> bool:
        return phrase in text

    def contains_word(text: str, word: str) -> bool:
        return re.search(rf"\b{re.escape(word)}\b", text) is not None

    for keyword in job_keywords:
        kw = keyword.strip()
        if not kw:
            continue
        kw_lower = kw.lower()
        kw_norm = normalize(kw)

        # Strict terms: require verbatim presence in resume_lower; special case for SAR
        if kw_norm in strict_terms:
            if kw_norm == 'synthetic aperture radar':
                if contains_phrase(resume_lower, 'synthetic aperture radar') or contains_word(resume_lower, 'sar'):
                    matched_keywords.append(keyword)
            elif contains_phrase(resume_lower, kw_norm):
                matched_keywords.append(keyword)
            continue

        # Non-strict equivalences
        if kw_norm in equivalence_map:
            variants = equivalence_map[kw_norm]
            if any((contains_phrase(resume_norm, normalize(v)) if ' ' in v else contains_word(resume_lower, v.lower())) for v in variants):
                matched_keywords.append(keyword)
            continue

        # Default conservative matching
        if ' ' in kw_norm:
            # Multi-word: match phrase in normalized resume
            if contains_phrase(resume_norm, kw_norm):
                matched_keywords.append(keyword)
        else:
            # Single word: word-boundary match; avoid very short tokens
            if len(kw_norm) >= 3 and contains_word(resume_lower, kw_norm):
                matched_keywords.append(keyword)

    return matched_keywords

def run_agentic_flow(job_description, resumes, openai_api_key):
    """
    Main function to process job description and resumes, returning cosine similarity scores.

    Args:
        job_description (str): Job description text
        resumes (list): List of resume files
        openai_api_key (str): OpenAI API key (for summary generation)

    Returns:
        list: List of candidate dictionaries with cosine similarity scores
    """
    clear_logs()
    log_agentic_flow_start()

    # Step 1: Extract technical keywords from job description
    log_step_header('EXTRACT_KEYWORDS', 'Extracting technical keywords from job description.')
    job_text = job_description.strip()
    # GPT will automatically extract relevant keywords from any job description
    keywords = extract_technical_keywords(job_text, openai_api_key)
    # Returns only what's relevant for candidate shortlisting
    log_technical_keywords(keywords)

    # Step 2: Read resumes
    resume_texts = []
    resume_names = []

    for file in resumes:
        text = extract_text_from_file(file)
        if text.strip():
            name_only = extract_candidate_name(file.name, text)
            resume_names.append(name_only)
            resume_texts.append(text)
        else:
            log_agent_thought('READ_RESUME', f'Failed to extract content from {file.name}.')

    log_resume_processing(resume_names)

    # Step 3: Generate embeddings and compute similarity
    log_step_header('EMBEDDING_AND_SIMILARITY', 'Generating embeddings and computing cosine similarity scores.')
    model = get_embedding_model()
    job_embedding = embed_text(job_text, model)
    job_chunks = generate_chunk_embeddings(job_text, model)

    # Calculate cosine similarity scores for each resume
    candidates = []
    for idx, (name, text) in enumerate(zip(resume_names, resume_texts)):
        resume_embedding = embed_text(text, model)
        cosine_score = calculate_cosine_similarity(job_embedding, resume_embedding)

        # Chunk-level max similarity (captures localized strong matches)
        resume_chunks = generate_chunk_embeddings(text, model)
        if job_chunks and resume_chunks:
            # Compute max cosine across chunk pairs using dot products (already L2-normalized)
            max_chunk_sim = 0.0
            for jv in job_chunks:
                for rv in resume_chunks:
                    sim = float(np.dot(np.array(jv), np.array(rv)))
                    if sim > max_chunk_sim:
                        max_chunk_sim = sim
        else:
            max_chunk_sim = cosine_score

        # Find keyword matches for this resume
        matched_keywords = find_keyword_matches(text, keywords)
        keyword_match_count = len(matched_keywords)

        # Generate grounded summary (strictly using resume evidence)
        summary = generate_fit_summary(job_text, text, matched_keywords)

        # Combined ranking score: weighted blend of document cosine, max chunk sim, and keyword coverage
        keyword_coverage = (keyword_match_count / max(len(keywords), 1)) if keywords else 0.0
        combined_score = 0.6 * float(cosine_score) + 0.3 * float(max_chunk_sim) + 0.1 * float(keyword_coverage)

        candidates.append({
            'name': name,
            'score': float(combined_score),
            'similarity': float(cosine_score),
            'max_chunk_similarity': float(max_chunk_sim),
            'keyword_coverage': float(keyword_coverage),
            'keyword_matches': matched_keywords,
            'keyword_count': keyword_match_count,
            'summary': summary
        })

    # Sort by cosine similarity score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    log_similarity_scores(candidates)
    log_agentic_flow_end(candidates[0])

    # Determine how many candidates to return/write
    # <=5 resumes -> show all; 6-9 resumes -> show all; >=10 -> cap at 10
    top_k = min(len(candidates), 10)

    # Write recommendations file
    with open(RECOMMEND_PATH, 'w', encoding='utf-8') as f:
        f.write('Top Recommended Candidates\n')
        f.write(f"{DIVIDER}\n")
        f.write(f"Technical Keywords: {', '.join(keywords)}\n")
        f.write(f"{DIVIDER}\n")
        for c in candidates[:top_k]:
            f.write(f"Name: {c['name']}\n")
            f.write(f"Overall Rank Score: {c['score']:.4f} | Cosine: {c['similarity']:.4f} | Max-Chunk: {c['max_chunk_similarity']:.4f} | Keyword Coverage: {c['keyword_coverage']:.2f}\n")
            f.write(f"Keyword Matches ({c['keyword_count']}): {', '.join(c['keyword_matches'])}\n")
            f.write(f"Summary: {c['summary']}\n")
            f.write(f"{DIVIDER}\n")

    return candidates[:top_k]