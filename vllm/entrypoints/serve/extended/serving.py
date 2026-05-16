# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from http import HTTPStatus

import torch

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.extended.protocol import (
    ABPairwiseRequest,
    ABPairwiseResponse,
    ABPairwiseResult,
    PerplexityRequest,
    PerplexityResponse,
)
from vllm.inputs import tokens_input
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

class _TargetTokenLogprobsProcessor:
    """
    Teacher-forcing LogitsProcessor for target-token-only scoring.

    At each decode step it:
      1. Records log P(target | context) = logit[target] - logsumexp(logits),
         computed from the *original* logits — O(vocab_size), not O(seq * vocab).
      2. Forces the sampler to emit the target token so the next step is
         conditioned on the correct context (teacher forcing).

    Because only the prefix is submitted as the vLLM prompt, the KV cache is
    naturally bounded to prefix_len and never reaches into the scored suffix.
    """

    def __init__(self, target_ids: list[int]) -> None:
        self._targets = target_ids
        self._step = 0
        self.logprobs: list[float] = []

    def __call__(self, token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if self._step >= len(self._targets):
            return logits
        target = self._targets[self._step]
        self._step += 1
        # Compute from the original distribution before any modification.
        log_prob = float(logits[target]) - float(torch.logsumexp(logits, dim=-1))
        self.logprobs.append(log_prob)
        # Force the sampler to emit the target token.
        forced = torch.full_like(logits, float("-inf"))
        forced[target] = logits[target]
        return forced


def _is_engine_fatal(exc: BaseException) -> bool:
    etype = type(exc).__name__
    if etype in ("EngineDeadError", "AsyncEngineDeadError", "OutOfMemoryError"):
        return True
    msg = str(exc).lower()
    return "engine is dead" in msg or "out of memory" in msg


class ExtendedServing(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        *,
        request_logger: RequestLogger | None,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
        )

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    async def create_perplexity(
        self,
        request: PerplexityRequest,
        raw_request: Request,
    ) -> PerplexityResponse | ErrorResponse:
        error = await self._check_model(request)
        if error is not None:
            return error

        if request.prefix_len < 1:
            return self.create_error_response(
                "prefix_len must be >= 1 (the prompt submitted to the engine must be non-empty)",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        n = len(request.candidates_tokens)
        if n == 0:
            return PerplexityResponse(scores=[], best_index=-1, cached_tokens=[])
        if n == 1:
            return PerplexityResponse(scores=[1.0], best_index=0, cached_tokens=[None])

        total_start = time.perf_counter()
        base_id = self._base_request_id(raw_request)

        scoring_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                self._score_candidate(
                    token_ids=tokens,
                    idx=i,
                    request_base_id=base_id,
                    prefix_len=request.prefix_len,
                    aggregation=request.aggregation,
                )
                for i, tokens in enumerate(request.candidates_tokens)
            ],
            return_exceptions=True,
        )
        scoring_s = time.perf_counter() - scoring_start

        scores: list[float] = []
        cached_tokens: list[int | None] = []
        for i, item in enumerate(results):
            if isinstance(item, BaseException):
                status = (
                    HTTPStatus.INTERNAL_SERVER_ERROR
                    if _is_engine_fatal(item)
                    else HTTPStatus.UNPROCESSABLE_ENTITY
                )
                return self.create_error_response(
                    f"Candidate {i} scoring failed: {item}",
                    err_type="InternalError",
                    status_code=status,
                )
            _, score, cached = item
            scores.append(score)
            cached_tokens.append(cached)

        best_index = max(range(n), key=lambda i: scores[i])
        total_s = time.perf_counter() - total_start

        return PerplexityResponse(
            scores=scores,
            best_index=best_index,
            cached_tokens=cached_tokens,
            profile={
                "candidate_scoring_s": round(scoring_s, 6),
                "total_s": round(total_s, 6),
            },
        )

    async def _score_candidate(
        self,
        token_ids: list[int],
        idx: int,
        request_base_id: str,
        prefix_len: int,
        aggregation: str,
    ) -> tuple[int, float, int | None]:
        """
        Score a candidate using teacher-forced decoding over the suffix.

        Only the prefix is submitted as the vLLM prompt, so the KV cache is
        naturally bounded to prefix_len and never reaches the scored suffix.
        Memory cost is O(vocab_size) per suffix token instead of
        O(full_seq_len * vocab_size) that prompt_logprobs would require.
        """
        suffix = token_ids[prefix_len:]
        if not suffix:
            raise ValueError(f"Candidate {idx}: no suffix tokens to score (prefix_len={prefix_len})")

        request_id = f"ppl-{request_base_id}-{idx}"
        collector = _TargetTokenLogprobsProcessor(suffix)

        sampling_params = SamplingParams(
            max_tokens=len(suffix),
            temperature=0.0,
            logits_processors=[collector],
            detokenize=False,
        )
        result_gen = self.engine_client.generate(
            tokens_input(token_ids[:prefix_len]), sampling_params, request_id
        )

        final_res = None
        async for res in result_gen:
            final_res = res

        if final_res is None or not final_res.outputs:
            raise RuntimeError(f"Engine returned no output for candidate {idx}")

        if len(collector.logprobs) != len(suffix):
            raise RuntimeError(
                f"Candidate {idx}: collected {len(collector.logprobs)} logprobs, "
                f"expected {len(suffix)} (prefix_len={prefix_len})"
            )

        score = (
            sum(collector.logprobs) / len(collector.logprobs)
            if aggregation == "mean"
            else sum(collector.logprobs)
        )
        return idx, score, final_res.num_cached_tokens

    # ------------------------------------------------------------------
    # AB Pairwise
    # ------------------------------------------------------------------

    async def create_ab_pairwise(
        self,
        request: ABPairwiseRequest,
        raw_request: Request,
    ) -> ABPairwiseResponse | ErrorResponse:
        error = await self._check_model(request)
        if error is not None:
            return error

        n = len(request.candidates_prompts)
        if n == 0:
            return ABPairwiseResponse(results=[])

        base_id = self._base_request_id(raw_request)
        total_start = time.perf_counter()

        scoring_start = time.perf_counter()
        raw_results = await asyncio.gather(
            *[
                self._run_pairwise_prompt(
                    token_ids=prompt,
                    idx=i,
                    base_id=base_id,
                    temperature=request.temperature,
                    seed=request.seed,
                    allowed_token_ids=request.allowed_token_ids,
                )
                for i, prompt in enumerate(request.candidates_prompts)
            ],
            return_exceptions=True,
        )
        scoring_s = time.perf_counter() - scoring_start

        tokenizer = self.renderer.get_tokenizer()
        results: list[ABPairwiseResult] = []
        for item in raw_results:
            if isinstance(item, BaseException):
                logger.warning("AB pairwise prompt failed: %s", item)
                results.append(ABPairwiseResult(token_ids=[], text=""))
            else:
                ids, cached = item
                text = tokenizer.decode(ids, skip_special_tokens=True)
                results.append(ABPairwiseResult(token_ids=ids, text=text, cached_tokens=cached))

        return ABPairwiseResponse(
            results=results,
            profile={
                "candidate_scoring_s": round(scoring_s, 6),
                "total_s": round(time.perf_counter() - total_start, 6),
            },
        )

    async def _run_pairwise_prompt(
        self,
        token_ids: list[int],
        idx: int,
        base_id: str,
        temperature: float,
        seed: int,
        allowed_token_ids: list[int] | None,
    ) -> tuple[list[int], int | None]:
        request_id = f"abpw-{base_id}-{idx}"
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=temperature,
            seed=seed + idx,
            detokenize=False,
            allowed_token_ids=allowed_token_ids,
            # prompt_logprobs is not used here, so the auto-set in
            # SamplingParams.__post_init__ leaves skip_reading_prefix_cache=False.
            # This allows the shared system-prompt / query prefix to be reused
            # from the KV cache across calls.
        )
        result_gen = self.engine_client.generate(
            tokens_input(token_ids), sampling_params, request_id
        )
        final_res = None
        async for res in result_gen:
            final_res = res

        if final_res is None or not final_res.outputs:
            raise RuntimeError(f"Engine returned no output for prompt {idx}")

        return list(final_res.outputs[0].token_ids), final_res.num_cached_tokens
