# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


class PerplexityRequest(OpenAIBaseModel):
    model: str
    # Number of prefix tokens excluded from the score (set to 0 to score all).
    prefix_len: int = 0
    # Pre-tokenized candidates. All lists must be non-empty.
    candidates_tokens: list[list[int]]
    aggregation: Literal["mean", "sum"] = "mean"


class PerplexityResponse(OpenAIBaseModel):
    # Log-likelihood scores (higher = better). One entry per candidate.
    scores: list[float]
    best_index: int
    # KV cache hits per candidate (None if not reported by engine).
    cached_tokens: list[int | None]
    profile: dict[str, float] = {}


class ABPairwiseResult(OpenAIBaseModel):
    token_ids: list[int]
    text: str
    # KV cache hits for this prompt (None if not reported by engine).
    cached_tokens: int | None = None


class ABPairwiseRequest(OpenAIBaseModel):
    model: str
    # One pre-built judge prompt per pair comparison (chat template applied by caller).
    candidates_prompts: list[list[int]]
    # Constrained decoding: only these token IDs can be sampled (shared across all prompts).
    allowed_token_ids: list[int] | None = None
    temperature: float = 0.0
    seed: int = 42042


class ABPairwiseResponse(OpenAIBaseModel):
    # One entry per input prompt, in the same order.
    results: list[ABPairwiseResult]
    profile: dict[str, float] = {}
