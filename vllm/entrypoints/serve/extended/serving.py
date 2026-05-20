# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
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
)
from vllm.inputs import tokens_input
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


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
