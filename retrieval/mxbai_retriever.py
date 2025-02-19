from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from tensorflow.python.ops.gen_nn_ops import TopK
import torch

from .base_retrieval import BaseRetriever

class MxbaiRetriever(BaseRetriever):
    def __init__(self, model_name, collection):
        self.model = SentenceTransformer(model_name)
        self.collection = collection

    def embed(self, query:List[str], **kwargs) -> np.ndarray:
        emb = self.model.encode( query, show_progress_bar=False, convert_to_numpy=True, **kwargs)
        return emb

    def retrieve(self, collection_name, query: List[str], topk=5) -> List[str]:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # query_emb = self.embed(query=query, batch_size=32, device=device)
        contexts = self.collection.query(query_texts=query, n_results=topk)
        return contexts
        