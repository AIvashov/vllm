# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


class ABPairwiseResult(OpenAIBaseModel):
    token_ids: list[int]
    text: str
    # KV cache hits for this prompt (None if not reported by engine).
    cached_tokens: int | None = None


class ABPairwiseRequest(OpenAIBaseModel):
    model: str
    # One pre-built judge prompt per pair comparison (chat template applied by caller).
    candidates_prompts: list[list[int]]
    # Constrained decoding: only these token IDs can be sampled (shared across
    # all prompts).
    allowed_token_ids: list[int] | None = None
    temperature: float = 0.0
    seed: int = 42042


class ABPairwiseResponse(OpenAIBaseModel):
    # One entry per input prompt, in the same order.
    results: list[ABPairwiseResult]
    profile: dict[str, float | int] = {}
