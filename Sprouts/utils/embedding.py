# utils/embedding.py

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Caching the model to avoid reloading it every time
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        # Force CPU to avoid meta/cuda device issues on some Windows/PyTorch combos
        _model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return _model

def _chunk_text_by_words(text: str, chunk_size: int = 256, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _encode_full_text(text: str, st_model: SentenceTransformer) -> np.ndarray:
    # Encode full text by chunking long inputs and averaging embeddings
    chunks = _chunk_text_by_words(text)
    if len(chunks) == 1:
        return st_model.encode(chunks[0], convert_to_numpy=True)
    embeddings = st_model.encode(chunks, convert_to_numpy=True)
    mean_vec = embeddings.mean(axis=0)
    # L2 normalize the mean vector for stable cosine similarity
    norm = np.linalg.norm(mean_vec) or 1.0
    return mean_vec / norm


def generate_chunk_embeddings(text: str, model) -> List[np.ndarray]:
    """
    Generate L2-normalized embeddings for each chunk of the input text.
    Returns a list of NumPy vectors.
    """
    st_model = get_embedding_model()
    chunks = _chunk_text_by_words(text)
    if not chunks:
        return []
    if len(chunks) == 1:
        vec = st_model.encode(chunks[0], convert_to_numpy=True)
        norm = np.linalg.norm(vec) or 1.0
        return [vec / norm]
    embs = st_model.encode(chunks, convert_to_numpy=True)
    # L2 normalize each row
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (embs / norms).tolist()


def generate_embedding(text, model, openai_api_key=None):
    if model == "openai":
        import openai
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-large",
            api_key=openai_api_key
        )
        return response['data'][0]['embedding']
    elif model == "baai":
        from transformers import AutoTokenizer, AutoModel
        import torch
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings
    else:
        st_model = get_embedding_model()
        """
        Generate an embedding for the given text using the provided model.
        Processes the entire document by chunking and averaging if needed.
        """
        return _encode_full_text(text, st_model)