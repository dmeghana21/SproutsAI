# utils/embedding.py

from sentence_transformers import SentenceTransformer

# Caching the model to avoid reloading it every time
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

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
        model = get_embedding_model()
        """
        Generate an embedding for the given text using the provided model.
        """
        return model.encode(text, convert_to_tensor=True)