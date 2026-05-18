# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from dataclasses import dataclass
from http import HTTPStatus

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


def _is_engine_fatal(exc: BaseException) -> bool:
    etype = type(exc).__name__
    if etype in ("EngineDeadError", "AsyncEngineDeadError", "OutOfMemoryError"):
        return True
    msg = str(exc).lower()
    return "engine is dead" in msg or "out of memory" in msg


@dataclass
class _CandidateScore:
    idx: int
    token_logprobs: list[float | None]
    missing_positions: list[int]
    cached_tokens: int | None


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

        if request.prefix_len < 0:
            return self.create_error_response(
                "prefix_len must be >= 0",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        n = len(request.candidates_tokens)
        if n == 0:
            return PerplexityResponse(scores=[], best_index=-1, cached_tokens=[])

        score_start = self._score_start(request.prefix_len)
        for i, token_ids in enumerate(request.candidates_tokens):
            if not token_ids:
                return self.create_error_response(
                    f"candidates_tokens[{i}] must be non-empty",
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            if score_start >= len(token_ids):
                return self.create_error_response(
                    f"candidates_tokens[{i}] has no tokens to score "
                    f"(prefix_len={request.prefix_len}, num_tokens={len(token_ids)})",
                    status_code=HTTPStatus.BAD_REQUEST,
                )

        total_start = time.perf_counter()
        base_id = self._base_request_id(raw_request)

        scoring_start = time.perf_counter()
        results: list[_CandidateScore | BaseException | None] = [None] * n
        peer_fill_positions = 0

        # Phase 1: score candidates in a prefix-friendly order. Lexicographic
        # sorting keeps shared prefixes adjacent and naturally places a shorter
        # prefix before its longer extensions, so later candidates can hit the
        # KV-cache while we recover cached prompt_logprobs from earlier peers.
        for i in self._prefix_cache_order(request.candidates_tokens):
            try:
                results[i] = await self._score_candidate(
                    token_ids=request.candidates_tokens[i],
                    idx=i,
                    request_base_id=base_id,
                    prefix_len=request.prefix_len,
                )
            except Exception as exc:
                results[i] = exc
            else:
                peer_fill_positions += self._fill_missing_from_peer_candidates(
                    results, request.candidates_tokens
                )

        # Phase 2: fill prompt_logprobs gaps created by KV-cache hits. We first
        # reuse any logprobs available from sibling candidates with identical
        # token prefixes. Then we recompute the shortest remaining missing
        # prefix, publish its logprobs to siblings, and repeat. This lets
        # candidates with longer shared prefixes extend the known results.
        fill_count = 0
        fill_start = time.perf_counter()
        while True:
            peer_fill_positions += self._fill_missing_from_peer_candidates(
                results, request.candidates_tokens
            )
            fill_targets = [
                (max(result.missing_positions), min(result.missing_positions), i)
                for i, result in enumerate(results)
                if isinstance(result, _CandidateScore) and result.missing_positions
            ]
            if not fill_targets:
                break

            fill_end, _, i = min(fill_targets)
            result = results[i]
            assert isinstance(result, _CandidateScore)
            fill_count += 1
            logger.info(
                "Filling %d missing prompt logprobs for candidate %d up to "
                "position %d (cached_tokens=%s)",
                len(result.missing_positions),
                i,
                fill_end,
                result.cached_tokens,
            )
            try:
                results[i] = await self._fill_missing_candidate_logprobs(
                    partial=result,
                    token_ids=request.candidates_tokens[i],
                    idx=i,
                    request_base_id=base_id,
                    prefix_len=request.prefix_len,
                )
            except Exception as exc:
                results[i] = exc
        fill_s = time.perf_counter() - fill_start

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
            try:
                score = self._aggregate_token_logprobs(
                    item.token_logprobs,
                    prefix_len=request.prefix_len,
                    aggregation=request.aggregation,
                )
            except Exception as exc:
                return self.create_error_response(
                    f"Candidate {i} scoring failed: {exc}",
                    err_type="InternalError",
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                )
            scores.append(score)
            cached_tokens.append(item.cached_tokens)

        best_index = max(range(n), key=lambda i: scores[i])
        total_s = time.perf_counter() - total_start

        return PerplexityResponse(
            scores=scores,
            best_index=best_index,
            cached_tokens=cached_tokens,
            profile={
                "candidate_scoring_s": round(scoring_s, 6),
                "peer_fill_positions": peer_fill_positions,
                "cache_fill_s": round(fill_s, 6),
                "cache_fill_requests": fill_count,
                "total_s": round(total_s, 6),
            },
        )

    @staticmethod
    def _prefix_cache_order(candidates_tokens: list[list[int]]) -> list[int]:
        return sorted(range(len(candidates_tokens)), key=lambda i: candidates_tokens[i])

    async def _score_candidate(
        self,
        token_ids: list[int],
        idx: int,
        request_base_id: str,
        prefix_len: int,
        skip_reading_prefix_cache: bool = False,
    ) -> _CandidateScore:
        """
        Score a candidate with a single engine request using prompt_logprobs=1.

        One request per candidate (not per token). Cached prompt positions are
        returned as missing prompt_logprobs and filled by the caller if they
        overlap the scored suffix.
        """
        request_id = f"ppl-{request_base_id}-{idx}"
        final_res = await self._run_prompt_logprobs_request(
            token_ids=token_ids,
            request_id=request_id,
            skip_reading_prefix_cache=skip_reading_prefix_cache,
        )
        token_logprobs, missing_positions = self._collect_candidate_logprobs(
            token_ids=token_ids,
            prompt_logprobs=final_res.prompt_logprobs,
            prefix_len=prefix_len,
        )
        return _CandidateScore(
            idx=idx,
            token_logprobs=token_logprobs,
            missing_positions=missing_positions,
            cached_tokens=final_res.num_cached_tokens,
        )

    async def _run_prompt_logprobs_request(
        self,
        token_ids: list[int],
        request_id: str,
        skip_reading_prefix_cache: bool,
    ):
        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            detokenize=False,
            skip_reading_prefix_cache=skip_reading_prefix_cache,
        )
        result_gen = self.engine_client.generate(
            tokens_input(token_ids), sampling_params, request_id
        )
        final_res = None
        async for res in result_gen:
            final_res = res

        if final_res is None or final_res.prompt_logprobs is None:
            raise RuntimeError(f"No prompt_logprobs returned for request {request_id}")

        return final_res

    @staticmethod
    def _score_start(prefix_len: int) -> int:
        # prompt_logprobs[0] is always None: the first prompt token has no
        # preceding context, so scoring all tokens starts from position 1.
        return max(prefix_len, 1)

    def _collect_candidate_logprobs(
        self,
        token_ids: list[int],
        prompt_logprobs,
        prefix_len: int,
    ) -> tuple[list[float | None], list[int]]:
        if len(prompt_logprobs) != len(token_ids):
            raise RuntimeError(
                f"Prompt logprobs length mismatch: got {len(prompt_logprobs)}, "
                f"expected {len(token_ids)}"
            )

        token_logprobs: list[float | None] = [None] * len(token_ids)
        missing_positions: list[int] = []
        for i in range(self._score_start(prefix_len), len(token_ids)):
            entry = prompt_logprobs[i]
            if not entry:
                missing_positions.append(i)
                continue
            lp = entry.get(token_ids[i])
            if lp is not None:
                token_logprobs[i] = lp.logprob
            else:
                missing_positions.append(i)
        return token_logprobs, missing_positions

    @staticmethod
    def _prefix_matches_through(
        left: list[int],
        right: list[int],
        pos: int,
    ) -> bool:
        return (
            len(left) > pos
            and len(right) > pos
            and left[: pos + 1] == right[: pos + 1]
        )

    def _fill_missing_from_peer_candidates(
        self,
        results: list,
        candidates_tokens: list[list[int]],
    ) -> int:
        candidate_results = [
            (i, result)
            for i, result in enumerate(results)
            if isinstance(result, _CandidateScore)
        ]
        if len(candidate_results) < 2:
            return 0

        num_filled = 0
        for i, result in candidate_results:
            if not result.missing_positions:
                continue

            remaining_positions: list[int] = []
            token_ids = candidates_tokens[i]
            for pos in result.missing_positions:
                peer_logprob = None
                for peer_i, peer_result in candidate_results:
                    if (
                        peer_i == i
                        or pos >= len(peer_result.token_logprobs)
                        or peer_result.token_logprobs[pos] is None
                    ):
                        continue
                    if self._prefix_matches_through(
                        token_ids, candidates_tokens[peer_i], pos
                    ):
                        peer_logprob = peer_result.token_logprobs[pos]
                        break

                if peer_logprob is None:
                    remaining_positions.append(pos)
                else:
                    result.token_logprobs[pos] = peer_logprob
                    num_filled += 1

            result.missing_positions = remaining_positions

        return num_filled

    async def _fill_missing_candidate_logprobs(
        self,
        partial: _CandidateScore,
        token_ids: list[int],
        idx: int,
        request_base_id: str,
        prefix_len: int,
    ) -> _CandidateScore:
        fill_end = max(partial.missing_positions)
        fill_token_ids = token_ids[: fill_end + 1]
        request_id = f"ppl-{request_base_id}-fill-{idx}-{fill_end}"
        final_res = await self._run_prompt_logprobs_request(
            token_ids=fill_token_ids,
            request_id=request_id,
            skip_reading_prefix_cache=True,
        )
        fill_logprobs, _ = self._collect_candidate_logprobs(
            token_ids=fill_token_ids,
            prompt_logprobs=final_res.prompt_logprobs,
            prefix_len=prefix_len,
        )

        still_missing: list[int] = []
        for pos in partial.missing_positions:
            logprob = fill_logprobs[pos]
            if logprob is None:
                still_missing.append(pos)
            else:
                partial.token_logprobs[pos] = logprob

        partial.missing_positions = still_missing
        if still_missing:
            missing_preview = still_missing[:8]
            raise RuntimeError(
                f"Candidate {idx}: incomplete prompt_logprobs after cache fill "
                f"(prefix_len={prefix_len}, total_tokens={len(token_ids)}, "
                f"missing_count={len(still_missing)}, "
                f"missing_preview={missing_preview})"
            )
        return partial

    def _aggregate_token_logprobs(
        self,
        token_logprobs: list[float | None],
        prefix_len: int,
        aggregation: str,
    ) -> float:
        logprobs = token_logprobs[self._score_start(prefix_len) :]
        if any(logprob is None for logprob in logprobs):
            missing_positions = [
                i
                for i, logprob in enumerate(token_logprobs)
                if i >= self._score_start(prefix_len) and logprob is None
            ]
            raise RuntimeError(
                "Cannot aggregate incomplete prompt_logprobs "
                f"(missing_count={len(missing_positions)}, "
                f"missing_preview={missing_positions[:8]})"
            )
        complete_logprobs = [logprob for logprob in logprobs if logprob is not None]
        return (
            sum(complete_logprobs) / len(complete_logprobs)
            if aggregation == "mean"
            else sum(complete_logprobs)
        )

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
                results.append(
                    ABPairwiseResult(token_ids=ids, text=text, cached_tokens=cached)
                )

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
