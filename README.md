# Candidate Recommendation Engine

This project is a take-home assignment for the Machine Learning Engineer Internship at **SproutsAI**. The goal is to build a web application that recommends the most relevant candidates for a job description based on semantic similarity between resume content and the job role, enhanced with AI-generated fit summaries.

---

## 💼 Features

- Accepts a **job description** via text input
- Accepts multiple **candidate resumes** via file upload (PDF or TXT)
- Extracts raw text from resumes using PyMuPDF
- Generates semantic embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`)
- Computes **cosine similarity** between job and each resume
- Ranks and displays the **top 5–10 most relevant candidates**
- Uses **OpenAI GPT-4o** to generate a short summary explaining why each candidate is a good fit

---

## 🛠️ Tech Stack

| Layer         | Tool / Library                |
|---------------|-------------------------------|
| Frontend      | Streamlit                     |
| Resume Parsing| PyMuPDF (`fitz`)              |
| Embeddings    | SentenceTransformers          |
| Similarity    | Scikit-learn (cosine similarity) |
| AI Summary    | OpenAI GPT-4o                 |
| Deployment    | Streamlit Cloud               |

---

## 🧪 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/candidate-recommender.git

### 2. Install dependencies
cd candidate-recommender

pip install -r requirements.txt

### 3. Set your OpenAI API key
Create a .env file in the root directory:
OPENAI_API_KEY=your-openai-api-key-here

4. Run the app
streamlit run app.py

### Project Structure

candidate_recommender/
├── app.py                  # Main Streamlit app
├── .env                    # OpenAI API key (not committed)
├── requirements.txt        # Python dependencies
├── utils/                  # Logic modules
│   ├── parser.py
│   ├── embedding.py
│   ├── similarity.py
│   └── gpt_summary.py
├── .streamlit/             # Streamlit config (optional)
│   └── config.toml
└── README.md               # This file


📌 Assumptions
Resume files are either .pdf or .txt.

The embedding model and OpenAI summary are both run locally via API, assuming stable internet.

🔗 Deployment
The app is deployed on Streamlit Cloud and accessible via:

https://your-streamlit-url.streamlit.app

