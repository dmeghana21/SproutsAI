import warnings
import streamlit as st
import os
from dotenv import load_dotenv
from utils.agent import run_agentic_flow, extract_technical_keywords

# Suppress PyTorch FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Candidate Recommendation Engine", layout="wide")
st.title("Candidate Recommendation Engine")
st.markdown("Upload resumes and paste a job description to find the most relevant candidates.")

# Get OpenAI API Key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

job_description = st.text_area("Enter Job Description", height=200)
uploaded_files = st.file_uploader(
    "Upload Candidate Resumes (PDF only)",
    type=["pdf"],
    accept_multiple_files=True
)

# Paths for logs and recommendations
log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'agent_thoughtlog.txt'))
rec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'recommended_candidates.txt'))

# Run recommendation and store results in session_state
if st.button("Recommend Candidates"):
    if not openai_api_key:
        st.warning("Please set your OpenAI API key in the environment variables.")
    elif not job_description:
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        with st.spinner("Processing..."):
            candidates = run_agentic_flow(job_description, uploaded_files, openai_api_key)
            st.session_state['candidates'] = candidates
            # Read and store logs and recommendations in session_state
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8") as f:
                    st.session_state['agent_log'] = f.read()
            if os.path.exists(rec_path):
                with open(rec_path, "r", encoding="utf-8") as f:
                    st.session_state['rec_file'] = f.read()

# Display recommended candidates if available
if 'candidates' in st.session_state:
    candidates = st.session_state['candidates']
    st.subheader(f"Top {len(candidates)} Recommended Candidate{'s' if len(candidates) > 1 else ''}")
    
    for rank, candidate in enumerate(candidates):
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### {rank+1}. {candidate['name']}")
                st.markdown(f"**AI Summary:** {candidate['summary']}")
                
                # Display keyword matches
                if 'keyword_matches' in candidate and candidate['keyword_matches']:
                    st.markdown(f"**Keyword Matches ({candidate['keyword_count']}):**")
                    # Display keywords in a more readable format
                    keyword_text = ", ".join(candidate['keyword_matches'])
                    st.markdown(f"*{keyword_text}*")
                else:
                    st.markdown("**Keyword Matches:** *None found*")
            
            with col2:
                st.markdown("**Score:**")
                if 'similarity' in candidate:
                    st.markdown(f"- **Cosine Similarity:** {candidate['similarity']:.4f}")
        
        st.markdown("---")

# Download buttons
col1, col2 = st.columns(2)
with col1:
    if 'rec_file' in st.session_state:
        st.download_button(
            label="Download Recommended Candidates",
            data=st.session_state['rec_file'],
            file_name="recommended_candidates.txt",
            mime="text/plain"
        )
with col2:
    if 'agent_log' in st.session_state:
        st.download_button(
            label="Download Agent Thought Log",
            data=st.session_state['agent_log'],
            file_name="agent_thoughtlog.txt",
            mime="text/plain"
        ) 