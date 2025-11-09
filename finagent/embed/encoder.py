from sentence_transformers import SentenceTransformer

class TextEncoder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)
