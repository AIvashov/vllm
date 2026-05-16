# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import math
import time

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

_NEG_INF = float("-inf")


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

        n = len(request.candidates_tokens)
        if n == 0:
            return PerplexityResponse(scores=[], best_index=-1, cached_tokens=[])
        if n == 1:
            return PerplexityResponse(scores=[1.0], best_index=0, cached_tokens=[None])

        total_start = time.perf_counter()

        scoring_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                self._score_candidate(
                    token_ids=tokens,
                    idx=i,
                    request_base_id=self._base_request_id(raw_request),
                    prefix_len=request.prefix_len,
                    aggregation=request.aggregation,
                )
                for i, tokens in enumerate(request.candidates_tokens)
            ],
            return_exceptions=True,
        )
        scoring_s = time.perf_counter() - scoring_start

        scores: list[float] = [_NEG_INF] * n
        cached_tokens: list[int | None] = [None] * n
        for item in results:
            if isinstance(item, BaseException):
                logger.warning("Perplexity candidate scoring failed: %s", item)
            else:
                idx, score, cached = item
                scores[idx] = score
                cached_tokens[idx] = cached

        finite = [i for i, s in enumerate(scores) if math.isfinite(s)]
        best_index = max(finite, key=lambda i: scores[i]) if finite else -1
        response_scores: list[float | None] = [
            s if math.isfinite(s) else None for s in scores
        ]
        total_s = time.perf_counter() - total_start

        return PerplexityResponse(
            scores=response_scores,
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
        skip_reading_prefix_cache: bool = False,
    ) -> tuple[int, float, int | None]:
        request_id = f"ppl-{request_base_id}-{idx}"
        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            detokenize=False,
            skip_reading_prefix_cache=skip_reading_prefix_cache,
        )
        engine_input = tokens_input(token_ids)
        result_gen = self.engine_client.generate(engine_input, sampling_params, request_id)

        final_res = None
        async for res in result_gen:
            final_res = res

        if final_res is None or final_res.prompt_logprobs is None:
            return idx, _NEG_INF, None

        cached = final_res.num_cached_tokens

        # If the KV cache reached past prefix_len, positions prefix_len..cached-1
        # have uninitialized logprob tensors (torch.empty). Retry without prefix
        # cache so the engine recomputes those positions with real logprobs.
        if not skip_reading_prefix_cache and cached is not None and cached > prefix_len:
            logger.warning(
                "Candidate %d: cached_tokens=%d > prefix_len=%d; retrying without prefix cache",
                idx, cached, prefix_len,
            )
            return await self._score_candidate(
                token_ids=token_ids,
                idx=idx,
                request_base_id=request_base_id + "-nocache",
                prefix_len=prefix_len,
                aggregation=aggregation,
                skip_reading_prefix_cache=True,
            )

        chunk_logprobs: list[float] = []
        for i, entry in enumerate(final_res.prompt_logprobs):
            if i < prefix_len or entry is None:
                continue
            # Look up the actual prompt token, not rank-1 (top-1 by probability).
            lp = entry.get(token_ids[i])
            if lp is not None:
                chunk_logprobs.append(lp.logprob)

        if not chunk_logprobs:
            return idx, _NEG_INF, cached

        score = (
            sum(chunk_logprobs) / len(chunk_logprobs)
            if aggregation == "mean"
            else sum(chunk_logprobs)
        )
        return idx, score, cached

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
