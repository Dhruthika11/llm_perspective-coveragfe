"""
Loads BAAI/bge-large-en-v1.5 via sentence-transformers.

Note on BGE usage convention: BGE models recommend prefixing *queries* (but
not documents/passages) with "Represent this sentence for searching relevant
passages: " when doing asymmetric retrieval. For our symmetric similarity
task (opinion vs opinion, not query vs document), we do NOT use that prefix -
we embed all texts (claims, human perspectives, LLM outputs) the same way,
since we're comparing opinions to opinions, not searching a corpus.
"""
from sentence_transformers import SentenceTransformer
import numpy as np


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings
