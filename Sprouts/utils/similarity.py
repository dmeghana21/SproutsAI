# utils/similarity.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

def compute_similarity_scores(job_embedding, resume_embeddings):
    """
    Compute cosine similarity scores between job_embedding and each resume embedding.
    
    Parameters:
        job_embedding (Tensor): Embedding vector for job description
        resume_embeddings (List[Tensor]): List of embedding vectors for resumes
    
    Returns:
        List[float]: Cosine similarity scores
    """
    job_vec = np.array(job_embedding)
    scores = []
    for emb in resume_embeddings:
        emb_vec = np.array(emb)
        score = np.dot(job_vec, emb_vec) / (np.linalg.norm(job_vec) * np.linalg.norm(emb_vec))
        scores.append(score)
    return scores