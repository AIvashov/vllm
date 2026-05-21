# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel
from vllm.entrypoints.serve.disagg.protocol import MultiModalFeatures


class ABPairwiseResult(OpenAIBaseModel):
    token_ids: list[int]
    text: str
    # KV cache hits for this prompt (None if not reported by engine).
    cached_tokens: int | None = None


class ABPairwiseRequest(OpenAIBaseModel):
    model: str
    # One pre-built judge prompt per pair comparison (chat template applied by caller).
    candidates_prompts: list[list[int]]
    # Optional multimodal features parallel to candidates_prompts.
    shared_features: MultiModalFeatures | None = None
    # Constrained decoding: only these token IDs can be sampled (shared across
    # all prompts).
    allowed_token_ids: list[int] | None = None
    temperature: float = 0.0
    seed: int = 42042
    cache_salt: str | None = None


class ABPairwiseResponse(OpenAIBaseModel):
    # One entry per input prompt, in the same order.
    results: list[ABPairwiseResult]
    profile: dict[str, float | int] = {}
