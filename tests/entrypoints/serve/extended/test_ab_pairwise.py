# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

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
    request: ABPairwiseRequest,
):
    serving = object.__new__(ExtendedServing)
    serving.engine_client = MagicMock()
    serving._check_model = MagicMock()
    serving._base_request_id = MagicMock(return_value="base")
    serving.renderer = MagicMock()
    serving.renderer.get_tokenizer.return_value.decode.return_value = "decoded"
    captured_engine_inputs = []

    async def no_error(*args, **kwargs):
        return None

    async def mock_generate(engine_input, *args, **kwargs):
        captured_engine_inputs.append(engine_input)
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[42])],
            num_cached_tokens=7,
        )

    serving._check_model.side_effect = no_error
    serving.engine_client.generate = MagicMock(side_effect=mock_generate)

    response = await serving.create_ab_pairwise(request, SimpleNamespace())

    return response, captured_engine_inputs


def test_ab_pairwise_request_accepts_legacy_payload():
    request = ABPairwiseRequest(
        model=MODEL_NAME,
        candidates_prompts=[[1, 2, 3]],
        allowed_token_ids=[10, 11],
    )

    assert request.shared_features is None


def test_ab_pairwise_request_accepts_shared_features():
    features = MultiModalFeatures(
        mm_hashes={"image": ["hash"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=0, length=2)]},
    )
    request = ABPairwiseRequest(
        model=MODEL_NAME,
        candidates_prompts=[[1, 2]],
        shared_features=features,
    )

    assert request.shared_features == features


@pytest.mark.asyncio
async def test_create_ab_pairwise_uses_tokens_input_for_text_only():
    response, engine_inputs = await _make_serving_and_run(
        request=ABPairwiseRequest(
            model=MODEL_NAME,
            candidates_prompts=[[1, 2, 3]],
            allowed_token_ids=[1, 2],
            cache_salt="salt",
        ),
    )

    assert response.results[0].token_ids == [42]
    assert response.results[0].cached_tokens == 7
    engine_input = engine_inputs[0]
    assert engine_input["type"] == "token"
    assert engine_input["prompt_token_ids"] == [1, 2, 3]
    assert engine_input["cache_salt"] == "salt"


@pytest.mark.asyncio
async def test_create_ab_pairwise_uses_mm_input_for_shared_features_cache_hit():
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

    _, engine_inputs = await _make_serving_and_run(
        request=ABPairwiseRequest(
            model=MODEL_NAME,
            candidates_prompts=[[1, 2, 3]],
            shared_features=features,
            allowed_token_ids=[1, 2],
            cache_salt="salt",
        ),
    )

    engine_input = engine_inputs[0]
    assert engine_input["type"] == "multimodal"
    assert engine_input["prompt_token_ids"] == [1, 2, 3]
    assert engine_input["cache_salt"] == "salt"
    assert engine_input["mm_kwargs"]["image"] == [None]
    placeholder = engine_input["mm_placeholders"]["image"][0]
    assert placeholder.is_embed is not None
    assert placeholder.is_embed.dtype == torch.bool
    assert placeholder.is_embed.tolist() == [False, True, False]


@pytest.mark.asyncio
async def test_create_ab_pairwise_decodes_shared_kwargs_data():
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

    _, engine_inputs = await _make_serving_and_run(
        request=ABPairwiseRequest(
            model=MODEL_NAME,
            candidates_prompts=[[1, 2], [3, 4]],
            shared_features=features,
        ),
    )

    engine_input = engine_inputs[0]
    decoded_item = engine_input["mm_kwargs"]["image"][0]
    assert decoded_item is not None
    assert torch.equal(decoded_item["pixel_values"].data, elem.data)
    assert len(engine_inputs) == 2
    assert engine_inputs[1]["prompt_token_ids"] == [3, 4]
    assert engine_inputs[1]["mm_kwargs"] is engine_input["mm_kwargs"]
    assert engine_inputs[1]["mm_placeholders"] is engine_input["mm_placeholders"]
