import os
import re
from typing import Optional
from utils.parser import extract_text_from_file
from utils.embedding import get_embedding_model, generate_embedding
from utils.similarity import compute_similarity_scores
from utils.gpt_summary import generate_fit_summary
from datetime import datetime
import openai

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOG_PATH = os.path.join(PROJECT_ROOT, 'logs', 'agent_thoughtlog.txt')
RECOMMEND_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'recommended_candidates.txt')

DIVIDER = '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'

JOB_TITLE_WORDS = {
    'developer', 'engineer', 'fullstack', 'full-stack', 'frontend', 'front-end', 'backend', 'back-end',
    'data', 'scientist', 'ml', 'ai', 'architect', 'lead', 'senior', 'junior', 'intern', 'manager',
    'consultant', 'analyst', 'software', 'devops', 'sdet', 'tester'
}

IGNORED_LINE_MARKERS = {
    'resume', 'curriculum vitae', 'cv', 'profile', 'summary', 'objective', 'experience', 'education',
    'linkedin', 'github', 'email', 'phone', 'contact'
}

PLACEHOLDER_TOKENS = {
    'first', 'surname', 'lastname', 'firstname', 'last', 'name', 'middle', 'given', 'family',
    'firstName'.lower(), 'lastName'.lower(), 'full', 'full name'
}

# Unicode letter-only token regex (excludes digits and underscore)
LETTER_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

def _format_first_last(tokens):
    first = tokens[0].title()
    last = tokens[-1].title() if len(tokens) > 1 else ''
    return f"{first} {last}".strip()

def _filter_name_tokens(tokens):
    filtered = [t for t in tokens if t and t.lower() not in PLACEHOLDER_TOKENS]
    return filtered

def extract_candidate_name_from_text(resume_text: str) -> Optional[str]:
    """
    Try to extract the candidate name from the resume text by scanning the top lines.
    Returns first+last formatted name if confidently found, else None.
    """
    if not resume_text:
        return None
    lines = [ln.strip() for ln in resume_text.splitlines()]
    # Scan only the first ~30 lines to find a clean name line
    for line in lines[:30]:
        if not line:
            continue
        low = line.lower()
        # Check for explicit 'name:' patterns
        if 'name:' in low:
            after = line.split(':', 1)[1].strip()
            tokens = _filter_name_tokens(LETTER_TOKEN_RE.findall(after))
            if len(tokens) >= 2 and not any(tok.lower() in JOB_TITLE_WORDS for tok in tokens):
                return _format_first_last(tokens)
            else:
                continue
        # skip lines with obvious non-name markers
        if any(marker in low for marker in IGNORED_LINE_MARKERS):
            continue
        if '@' in line or any(ch.isdigit() for ch in line):
            continue
        # allow letters (including unicode) and filter placeholders/job titles
        tokens = _filter_name_tokens(LETTER_TOKEN_RE.findall(line))
        if len(tokens) >= 2 and 2 <= len(tokens) <= 4:
            if any(tok.lower() in JOB_TITLE_WORDS for tok in tokens):
                continue
            return _format_first_last(tokens)
    return None

def extract_candidate_name(filename):
    """
    Extract and format candidate name from filename.
    Returns FirstName LastName style.
    """
    # Remove file extension and normalize separators
    base = os.path.splitext(filename)[0]
    base = re.sub(r'[_.-]+', ' ', base)
    # Lowercase for cleanup
    cleaned = base.lower()
    # Remove common non-name words and role markers and placeholders
    remove_words = {'resume', 'cv', 'application', 'cover', 'letter', 'profile'} | JOB_TITLE_WORDS | PLACEHOLDER_TOKENS
    cleaned = ' '.join(w for w in cleaned.split() if w and w not in remove_words)
    # Tokenize letters only (unicode-aware) and filter placeholders again
    tokens = _filter_name_tokens(LETTER_TOKEN_RE.findall(cleaned))
    if len(tokens) >= 2:
        return _format_first_last(tokens)
    if len(tokens) == 1:
        return tokens[0].title()
    # Fallback to title-cased base if nothing else
    return base.title()

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
            model="gpt-4",
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
    Find which keywords from the job description match in the resume.
    Searches through the complete resume text for comprehensive matching.

    Args:
        resume_text (str): Resume text
        job_keywords (list): Keywords extracted from job description

    Returns:
        list: List of keywords that match in the resume
    """
    resume_lower = resume_text.lower()
    matched_keywords = []

    for keyword in job_keywords:
        keyword_lower = keyword.lower().strip()

        # Check for exact match in complete resume
        if keyword_lower in resume_lower:
            matched_keywords.append(keyword)
        # Check for word boundary matches (more accurate)
        elif ' ' in keyword_lower:
            words = keyword_lower.split()
            # Check if all words from compound keyword exist in resume
            if all(word in resume_lower for word in words):
                matched_keywords.append(keyword)
        # Check for variations and abbreviations
        else:
            # Handle common variations
            variations = [keyword_lower]
            if keyword_lower == 'python':
                variations.extend(['py', 'python3', 'python2'])
            elif keyword_lower == 'javascript':
                variations.extend(['js', 'node.js', 'nodejs'])
            elif keyword_lower == 'machine learning':
                variations.extend(['ml', 'ai', 'artificial intelligence'])
            elif keyword_lower == 'api':
                variations.extend(['apis', 'rest api', 'graphql'])
            elif keyword_lower == 'sql':
                variations.extend(['mysql', 'postgresql', 'sqlite', 'database'])

            # Check if any variation exists in resume
            for variation in variations:
                if variation in resume_lower:
                    matched_keywords.append(keyword)
                    break

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
            name_from_text = extract_candidate_name_from_text(text)
            name_only = name_from_text if name_from_text else extract_candidate_name(file.name)
            resume_names.append(name_only)
            resume_texts.append(text)
        else:
            log_agent_thought('READ_RESUME', f'Failed to extract content from {file.name}.')

    log_resume_processing(resume_names)

    # Step 3: Generate embeddings and compute similarity
    log_step_header('EMBEDDING_AND_SIMILARITY', 'Generating embeddings and computing cosine similarity scores.')
    model = get_embedding_model()
    job_embedding = embed_text(job_text, model)

    # Calculate cosine similarity scores for each resume
    candidates = []
    for idx, (name, text) in enumerate(zip(resume_names, resume_texts)):
        resume_embedding = embed_text(text, model)
        cosine_score = calculate_cosine_similarity(job_embedding, resume_embedding)

        # Find keyword matches for this resume
        matched_keywords = find_keyword_matches(text, keywords)
        keyword_match_count = len(matched_keywords)

        # Generate summary
        summary = generate_fit_summary(job_text, text)

        candidates.append({
            'name': name,
            'score': float(cosine_score),
            'similarity': float(cosine_score),
            'keyword_matches': matched_keywords,
            'keyword_count': keyword_match_count,
            'summary': summary
        })

    # Sort by cosine similarity score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    log_similarity_scores(candidates)
    log_agentic_flow_end(candidates[0])

    # Determine how many candidates to return: clamp between 5 and 10 based on resume count
    total_candidates = len(candidates)
    top_n = max(5, min(10, total_candidates))

    # Write recommendations file
    with open(RECOMMEND_PATH, 'w', encoding='utf-8') as f:
        f.write(f'Top {top_n} Recommended Candidates\n')
        f.write(f"{DIVIDER}\n")
        f.write(f"Technical Keywords: {', '.join(keywords)}\n")
        f.write(f"{DIVIDER}\n")
        for c in candidates[:top_n]:
            f.write(f"Name: {c['name']}\n")
            f.write(f"Cosine Similarity Score: {c['similarity']:.4f}\n")
            f.write(f"Keyword Matches ({c['keyword_count']}): {', '.join(c['keyword_matches'])}\n")
            f.write(f"Summary: {c['summary']}\n")
            f.write(f"{DIVIDER}\n")

    return candidates[:top_n]