# utils/parser.py

import fitz  # PyMuPDF

def extract_text_from_file(file):
    """
    Extract text from an uploaded PDF or TXT file.
    """
    filename = file.name.lower()

    if filename.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
        return text

    elif filename.endswith(".pdf"):
        try:
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                return text
        except Exception as e:
            print(f"Error reading PDF {filename}: {e}")
            return ""

    else:
        return ""