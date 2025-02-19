from sentence_transformers import CrossEncoder
from typing import List
import numpy as np

from .base_reranker import BaseReranker

class MxbaiReranker(BaseReranker):

    def __init__(self, model_name):
        self.model = CrossEncoder(model_name)
    
    def rank(self, query: str, contexts: List[str], topk:int) -> List[str]:
        topk = min(topk, len(contexts))
        pairs = [(query, context) for context in contexts]
        scores = self.model.predict(pairs)
        indices = np.argsort(scores)[::-1]
        return [contexts[i] for i in indices][:topk]