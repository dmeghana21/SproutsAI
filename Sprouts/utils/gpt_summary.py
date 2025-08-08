# utils/gpt_summary.py

import os
from dotenv import load_dotenv
import openai

# Load environment variable
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # New client-based syntax

def generate_fit_summary(job_description, resume_text):
    """
    Generate a 2–3 line summary explaining why the candidate is a good fit for the role.
    """

    if not client.api_key:
        return "OpenAI API key not found."

    prompt = f"""
You are an AI assistant helping a recruiter. Based on the following job description and candidate resume,
generate a clear 2–3 sentence summary explaining why this person is a strong fit for the role.

Job Description:
{job_description}

Candidate Resume:
{resume_text}

Summary:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating summary: {str(e)}"