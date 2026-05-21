# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError

from vllm.entrypoints.serve.disagg.mm_serde import encode_mm_kwargs_item
from vllm.entrypoints.serve.disagg.protocol import (
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.serve.extended.protocol import ABPairwiseRequest
from vllm.entrypoints.serve.extended.serving import ExtendedServing
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalKwargsItem,
)

MODEL_NAME = "openai-community/gpt2"


async def _make_serving_and_run(
    *,
    token_ids: list[int],
    features: MultiModalFeatures | None,
):
    serving = object.__new__(ExtendedServing)
    serving.engine_client = MagicMock()
    captured_engine_inputs = []

    async def mock_generate(engine_input, *args, **kwargs):
        captured_engine_inputs.append(engine_input)
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[42])],
            num_cached_tokens=7,
        )

    serving.engine_client.generate = MagicMock(side_effect=mock_generate)

    result = await serving._run_pairwise_prompt(
        token_ids=token_ids,
        features=features,
        idx=0,
        base_id="base",
        temperature=0.0,
        seed=42042,
        allowed_token_ids=[1, 2],
        cache_salt="salt",
    )

    return result, captured_engine_inputs[0]


def test_ab_pairwise_request_accepts_legacy_payload():
    request = ABPairwiseRequest(
        model=MODEL_NAME,
        candidates_prompts=[[1, 2, 3]],
        allowed_token_ids=[10, 11],
    )

    assert request.candidates_features is None


def test_ab_pairwise_request_rejects_mismatched_features_length():
    with pytest.raises(ValidationError, match="candidates_features length"):
        ABPairwiseRequest(
            model=MODEL_NAME,
            candidates_prompts=[[1], [2]],
            candidates_features=[None],
        )


@pytest.mark.asyncio
async def test_run_pairwise_prompt_uses_tokens_input_for_text_only():
    result, engine_input = await _make_serving_and_run(
        token_ids=[1, 2, 3],
        features=None,
    )

    assert result == ([42], 7)
    assert engine_input["type"] == "token"
    assert engine_input["prompt_token_ids"] == [1, 2, 3]
    assert engine_input["cache_salt"] == "salt"


@pytest.mark.asyncio
async def test_run_pairwise_prompt_uses_mm_input_for_features_cache_hit():
    features = MultiModalFeatures(
        mm_hashes={"image": ["hash"]},
        mm_placeholders={
            "image": [
                PlaceholderRangeInfo(
                    offset=1,
                    length=3,
                    is_embed=[False, True, False],
                )
            ]
        },
    )

    _, engine_input = await _make_serving_and_run(
        token_ids=[1, 2, 3],
        features=features,
    )

    assert engine_input["type"] == "multimodal"
    assert engine_input["prompt_token_ids"] == [1, 2, 3]
    assert engine_input["mm_kwargs"]["image"] == [None]
    placeholder = engine_input["mm_placeholders"]["image"][0]
    assert placeholder.is_embed is not None
    assert placeholder.is_embed.dtype == torch.bool
    assert placeholder.is_embed.tolist() == [False, True, False]


@pytest.mark.asyncio
async def test_run_pairwise_prompt_decodes_kwargs_data():
    elem = MultiModalFieldElem(
        data=torch.ones(2, dtype=torch.float32),
        field=MultiModalBatchedField(),
    )
    item = MultiModalKwargsItem({"pixel_values": elem})
    features = MultiModalFeatures(
        mm_hashes={"image": ["hash"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=0, length=2)]},
        kwargs_data={"image": [encode_mm_kwargs_item(item)]},
    )

    _, engine_input = await _make_serving_and_run(
        token_ids=[1, 2],
        features=features,
    )

    decoded_item = engine_input["mm_kwargs"]["image"][0]
    assert decoded_item is not None
    assert torch.equal(decoded_item["pixel_values"].data, elem.data)
