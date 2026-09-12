from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import torch
from torch import nn

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class SharedLanguageEncoder(nn.Module):
    """Lightweight text encoder shared by driving and robotics."""

    def __init__(
        self,
        output_dim: int = 32,
        vocab_size: int = 512,
        token_dim: int = 32,
        max_tokens: int = 8,
    ) -> None:
        super().__init__()

        if output_dim <= 0:
            raise ValueError("output_dim must be greater than zero")

        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

        if token_dim <= 0:
            raise ValueError("token_dim must be greater than zero")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        self._output_dim = output_dim
        self._vocab_size = vocab_size
        self._token_dim = token_dim
        self._max_tokens = max_tokens

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=token_dim,
            padding_idx=0,
        )

        self.projection = nn.Sequential(
            nn.Linear(token_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    @property
    def output_dim(self) -> int:
        """Return the final language embedding size."""
        return self._output_dim

    @property
    def vocab_size(self) -> int:
        """Return the hashing vocabulary size."""
        return self._vocab_size

    @property
    def token_dim(self) -> int:
        """Return the token embedding dimension."""
        return self._token_dim

    @property
    def max_tokens(self) -> int:
        """Return the maximum number of tokens per instruction."""
        return self._max_tokens

    def tokenize(
        self,
        texts: Sequence[str],
    ) -> torch.Tensor:
        """Convert text instructions into deterministic token IDs."""
        if not texts:
            raise ValueError("texts must not be empty")

        token_rows: list[list[int]] = []

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("every language instruction must be a string")

            normalized = text.strip().lower()

            if not normalized:
                raise ValueError("language instruction must not be empty")

            tokens = _TOKEN_PATTERN.findall(normalized)

            if not tokens:
                raise ValueError(
                    "language instruction must contain alphanumeric tokens"
                )

            token_ids = [
                self._hash_token(token) for token in tokens[: self._max_tokens]
            ]

            padding = self._max_tokens - len(token_ids)
            token_ids.extend([0] * padding)

            token_rows.append(token_ids)

        return torch.tensor(
            token_rows,
            dtype=torch.long,
        )

    def forward(
        self,
        texts: Sequence[str],
    ) -> torch.Tensor:
        """Encode a batch of text instructions."""
        token_ids = self.tokenize(texts)

        device = self.embedding.weight.device
        token_ids = token_ids.to(device)

        token_embeddings = self.embedding(token_ids)

        mask = (token_ids != 0).unsqueeze(-1).to(token_embeddings.dtype)

        summed = (token_embeddings * mask).sum(dim=1)

        counts = mask.sum(dim=1).clamp_min(1.0)

        pooled = summed / counts

        return self.projection(pooled)

    def _hash_token(
        self,
        token: str,
    ) -> int:
        """Map a token deterministically into the local vocabulary."""
        digest = hashlib.sha256(token.encode("utf-8")).digest()

        integer = int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )

        return 1 + (integer % (self._vocab_size - 1))
