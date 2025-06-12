"""
Requires:
  transformers >= 4.51.0
  sentence-transformers >= 2.7.0
  inferless           (runtime)

Tip: for fastest load-time you may snapshot only the model weights:
  HF_HUB_ENABLE_HF_TRANSFER=1
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from typing import List, Optional
from pydantic import BaseModel, Field

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import torch
import inferless


# ---------- Request / Response schema ---------- #

@inferless.request
class RequestObjects(BaseModel):
    # Lists of strings to embed
    queries: List[str] = Field(
        default=["What is the capital of China?"],
        description="User queries (will receive the 'query' prompt by default)."
    )
    documents: List[str] = Field(
        default=["The capital of China is Beijing."],
        description="Reference documents to embed."
    )
    # Optional behaviour switches
    compute_similarity: Optional[bool] = Field(
        default=False,
        description="Return cosine-similarity matrix query × document."
    )
    # Advanced / rarely changed options
    prompt_name: Optional[str] = Field(
        default="query",
        description="Prompt key in model.prompts to use for queries."
    )
    normalize_embeddings: Optional[bool] = Field(
        default=True,
        description="Apply L2-normalisation to embeddings (recommended for cosine-sim)."
    )


@inferless.response
class ResponseObjects(BaseModel):
    # Returned as nested python lists to stay json-serialisable
    query_embeddings: List[List[float]]
    document_embeddings: List[List[float]]
    similarity: Optional[List[List[float]]] = None


# ---------- Inferless model ---------- #

class InferlessPythonModel:
    def initialize(self, context=None):
        """
        Downloads the model snapshot once on cold-start and keeps the
        SentenceTransformer object on GPU (if available).
        """
        self.model_id = "Qwen/Qwen3-Embedding-0.6B"

        # Only grab safetensors to trim size
        snapshot_download(repo_id=self.model_id, allow_patterns=["*.safetensors", "*.json", "tokenizer*"])

        # Enable FlashAttention 2 & left padding for speed (A100/H100/etc.)
        self.model = SentenceTransformer(
            self.model_id,
            model_kwargs={"attn_implementation": "flash_attention_2",
                          "device_map": "cuda" if torch.cuda.is_available() else "cpu"},
            tokenizer_kwargs={"padding_side": "left"}
        )

    def infer(self, request: RequestObjects) -> ResponseObjects:
        """
        1) Encodes queries and documents to embeddings.
        2) Optionally computes cosine similarity.
        3) Returns everything as JSON-serialisable python lists.
        """

        # --- Encode queries (with prompt) & documents --- #
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

        # --- Optional cosine similarity --- #
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
        """Optional clean-up on container shutdown."""
        self.model = None
