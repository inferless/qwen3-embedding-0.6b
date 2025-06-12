import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from typing import List, Optional
from pydantic import BaseModel, Field

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import torch
import inferless


@inferless.request
class RequestObjects(BaseModel):
    query: str = Field(default="What is the capital of China?")
    document: str = Field(default="The capital of China is Beijing.")
    compute_similarity: Optional[bool] = False
    prompt_name: Optional[str] = "query"
    normalize_embeddings: Optional[bool] = True

@inferless.response
class ResponseObjects(BaseModel):
    query_embeddings: List[float] = Field(default="Test output")
    document_embeddings: List[float] = Field(default="Test output")
    similarity: Optional[float] = 0.7897

class InferlessPythonModel:
    def initialize(self, context=None):
        self.model_id = "Qwen/Qwen3-Embedding-0.6B"
        snapshot_download(repo_id=self.model_id, allow_patterns=["*.safetensors", "*.json", "tokenizer*"])
        self.model = SentenceTransformer(
            self.model_id,
            model_kwargs={"device_map": "cuda"},
            tokenizer_kwargs={"padding_side": "left"}
        )

    def infer(self, request: RequestObjects) -> ResponseObjects:
        q_emb = self.model.encode(
            request.query,
            prompt_name=request.prompt_name,
            normalize_embeddings=request.normalize_embeddings,
            convert_to_numpy=True
        )
        d_emb = self.model.encode(
            request.document,
            normalize_embeddings=request.normalize_embeddings,
            convert_to_numpy=True
        )
        
        result = {}
        result["query_embeddings"] = q_emb.tolist()
        result["document_embeddings"] = d_emb.tolist()
        
        if request.compute_similarity:
            sim_tensor = self.model.similarity(
                torch.from_numpy(q_emb).to(self.model.device),
                torch.from_numpy(d_emb).to(self.model.device)
            )
            result["similarity"] = sim_tensor.cpu().tolist()[0][0]
            
        return ResponseObjects(**result)
        
    def finalize(self):
        self.model = None
