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
    similarity: List[float]= Field(default="Test output")


class InferlessPythonModel:
    def initialize(self, context=None):
        self.model_id = "Qwen/Qwen3-Embedding-0.6B"
        snapshot_download(repo_id=self.model_id, allow_patterns=["*.safetensors", "*.json", "tokenizer*"])
        self.model = SentenceTransformer(
            self.model_id,
            model_kwargs={"attn_implementation": "flash_attention_2",
                          "device_map": "cuda" if torch.cuda.is_available() else "cpu"},
            tokenizer_kwargs={"padding_side": "left"}
        )

    def infer(self, request: RequestObjects) -> ResponseObjects:
        q_emb = self.model.encode(
            request.queries,
            prompt_name=request.prompt_name,
            normalize_embeddings=request.normalize_embeddings,
            convert_to_numpy=True
        )
        d_emb = self.model.encode(
            request.documents,
            normalize_embeddings=request.normalize_embeddings,
            convert_to_numpy=True
        )
        sim_matrix = None
        if request.compute_similarity:
            # SentenceTransformers has a built-in helper
            # (returns torch tensor on the same device)
            sim_tensor = self.model.similarity(
                torch.from_numpy(q_emb).to(self.model.device),
                torch.from_numpy(d_emb).to(self.model.device)
            )
            sim_matrix = sim_tensor.cpu().tolist()

        # --- Build response --- #
        response = ResponseObjects(
            query_embeddings=q_emb.tolist(),
            document_embeddings=d_emb.tolist(),
            similarity=sim_matrix
        )
        return response

    def finalize(self):
        self.model = None
