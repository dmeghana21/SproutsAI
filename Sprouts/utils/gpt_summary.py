# utils/gpt_summary.py

import os
from dotenv import load_dotenv
import openai

# Load environment variable
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # New client-based syntax

def generate_fit_summary(job_description, resume_text, matched_keywords=None):
    """
    Generate a 4–5 sentence summary strictly grounded in resume evidence and mapped to the job description.
    - Use only resume facts (skills, tools, domains, employers, titles, dates, education, achievements).
    - Do not infer or assume anything beyond what is explicitly present in the resume.
    - If JD areas are not in the resume, do not claim them; you may note clearly transferable skills.
    - Maintain a concise, professional tone; avoid hype words unless verbatim in the resume.
    """

    if not client.api_key:
        return "OpenAI API key not found."

    # Build strict system and user instructions to reduce hallucinations
    system_msg = (
        "You are an ATS/job-match summarizer. Your task: given a JOB_DESCRIPTION and a RESUME_TEXT, "
        "produce a 4–5 sentence summary of how the candidate’s experience matches the job.\n\n"
        "DO NOT HALLUCINATE — do not add, assume, or imply skills, domains, tools, or experiences not explicitly present in the resume.\n\n"
        "Rules\n"
        "- Only mention skills, tools, domains, employers, titles, education, achievements, and years that are verbatim or explicitly stated in the resume.\n"
        "- If the job description contains terms not present in the resume (e.g., synthetic aperture radar, remote sensing, Intelligence Community, TS/SCI clearance, Databricks, geospatial), do not claim direct experience.\n"
        "- If relevant, you may say “experience in X may be transferable to Y” only if X appears in the resume and Y is in the JD.\n"
        "- Do not create or guess metrics, years, or achievements not in the resume.\n"
        "- Avoid hype terms like “expert”, “proven”, “recognized” unless they are in the resume.\n\n"
        "Output format\n"
        "- A single paragraph (4–5 sentences) containing only resume-backed facts.\n\n"
        "Checklist before finalizing\n"
        "- Every skill/tool/domain mentioned exists verbatim in the resume.\n"
        "- No invented achievements, metrics, or job duties.\n"
        "- If referencing a JD-only term, clearly mark it as ‘transferable.’\n"
    )

    matched_kw_text = (
        f"Matched keywords (from resume): {', '.join(matched_keywords)}\n\n"
        if matched_keywords else ""
    )

    user_msg = (
        "Inputs\n"
        "RESUME_TEXT (extracted from PDF):\n" + resume_text + "\n\n"
        "JOB_DESCRIPTION:\n" + job_description + "\n\n" +
        matched_kw_text +
        "Task\n"
        "Write a single-paragraph summary of exactly 4–5 sentences on how the candidate matches the job, "
        "using only facts explicitly in RESUME_TEXT. If JD areas are not covered in RESUME_TEXT, do not claim them; "
        "you may add a clearly labeled 'transferable.' sentence at the end only if appropriate."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=180,
        )
        text = response.choices[0].message.content.strip()
        # Best-effort cap to 5 sentences without adding content
        try:
            import re
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if len(sentences) > 5:
                text = " ".join(sentences[:5]).strip()
        except Exception:
            pass
        return text

    except Exception as e:
        return f"Error generating summary: {str(e)}"