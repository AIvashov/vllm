# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.extended.protocol import (
    ABPairwiseRequest,
    PerplexityRequest,
)
from vllm.entrypoints.serve.extended.serving import ExtendedServing
from vllm.entrypoints.utils import with_cancellation
from vllm.logger import init_logger

logger = init_logger(__name__)


def extended(request: Request) -> ExtendedServing:
    # Lazy init: ExtendedServing is lightweight (no async work needed), so we
    # create it on the first request rather than requiring a dedicated startup hook.
    if not hasattr(request.app.state, "extended_serving"):
        request.app.state.extended_serving = ExtendedServing(
            request.app.state.engine_client,
            request.app.state.openai_serving_models,
            request_logger=None,
        )
    return request.app.state.extended_serving


def attach_router(app: FastAPI) -> None:
    app.include_router(router)


router = APIRouter()

_ERROR_RESPONSES = {
    HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
    HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
    HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
}


@router.post(
    "/extended/v1/perplexity",
    dependencies=[Depends(validate_json_request)],
    responses=_ERROR_RESPONSES,
)
@with_cancellation
async def perplexity(request: PerplexityRequest, raw_request: Request):
    handler = extended(raw_request)
    result = await handler.create_perplexity(request, raw_request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    return JSONResponse(content=result.model_dump())


@router.post(
    "/extended/v1/ab_pairwise",
    dependencies=[Depends(validate_json_request)],
    responses=_ERROR_RESPONSES,
)
@with_cancellation
async def ab_pairwise(request: ABPairwiseRequest, raw_request: Request):
    handler = extended(raw_request)
    result = await handler.create_ab_pairwise(request, raw_request)
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    return JSONResponse(content=result.model_dump())
