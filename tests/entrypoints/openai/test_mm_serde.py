# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Roundtrip tests for multimodal serde used by the disagg generate endpoint."""

import pytest
import torch
from pydantic import ValidationError

from vllm.entrypoints.serve.disagg.mm_serde import (
    decode_mm_kwargs_item,
    encode_mm_kwargs_item,
)
from vllm.entrypoints.serve.disagg.protocol import (
    MultiModalFeatures,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)


def test_placeholder_range_info_is_embed_roundtrip():
    """PlaceholderRangeInfo preserves optional embed masks through JSON mode."""
    info = PlaceholderRangeInfo(
        offset=1,
        length=3,
        is_embed=[False, True, False],
    )

    payload = info.model_dump(mode="json")
    restored = PlaceholderRangeInfo.model_validate(payload)

    assert payload == {
        "offset": 1,
        "length": 3,
        "is_embed": [False, True, False],
    }
    assert restored.is_embed == [False, True, False]


def test_placeholder_range_info_accepts_legacy_payload_without_is_embed():
    """Old /generate payloads omit is_embed and keep whole-range semantics."""
    restored = PlaceholderRangeInfo.model_validate({"offset": 1, "length": 3})

    assert restored.is_embed is None


def test_placeholder_range_info_rejects_mismatched_is_embed_length():
    with pytest.raises(ValidationError, match="is_embed length must match"):
        PlaceholderRangeInfo(offset=1, length=3, is_embed=[True, False])


def test_extract_mm_features_serializes_placeholder_is_embed_mask():
    """Render features expose Gemma4-style sparse embed masks as JSON bools."""
    engine_input = {
        "type": "multimodal",
        "prompt_token_ids": [1, 2, 3],
        "mm_hashes": {"image": ["hash"]},
        "mm_placeholders": {
            "image": [
                PlaceholderRange(
                    offset=0,
                    length=3,
                    is_embed=torch.tensor([False, True, False]),
                )
            ]
        },
    }

    features = OpenAIServingRender._extract_mm_features(engine_input)

    assert features is not None
    placeholder = features.mm_placeholders["image"][0]
    assert placeholder.is_embed == [False, True, False]
    assert features.model_dump(mode="json")["mm_placeholders"]["image"][0][
        "is_embed"
    ] == [False, True, False]


def test_mm_kwargs_item_roundtrip():
    """Full roundtrip test with all three field types and multiple dtypes."""
    e1 = MultiModalFieldElem(
        data=torch.zeros(1000, dtype=torch.bfloat16),
        field=MultiModalBatchedField(),
    )
    e2 = MultiModalFieldElem(
        data=torch.ones(100, dtype=torch.int32),
        field=MultiModalSharedField(batch_size=4),
    )
    e3 = MultiModalFieldElem(
        data=torch.randn(20, dtype=torch.float32),
        field=MultiModalFlatField(slices=[slice(0, 10), slice(10, 20)], dim=0),
    )

    item = MultiModalKwargsItem({"pixel_values": e1, "grid_thw": e2, "embeds": e3})
    encoded = encode_mm_kwargs_item(item)

    # Encoded result is a base64 string
    assert isinstance(encoded, str)

    decoded = decode_mm_kwargs_item(encoded)

    assert set(decoded.keys()) == {"pixel_values", "grid_thw", "embeds"}
    assert torch.equal(item["pixel_values"].data, decoded["pixel_values"].data)
    assert torch.equal(item["grid_thw"].data, decoded["grid_thw"].data)
    assert torch.equal(item["embeds"].data, decoded["embeds"].data)
    assert isinstance(decoded["pixel_values"].field, MultiModalBatchedField)
    assert isinstance(decoded["grid_thw"].field, MultiModalSharedField)
    assert isinstance(decoded["embeds"].field, MultiModalFlatField)


def test_mm_kwargs_item_none_data():
    """Roundtrip with None data field."""
    elem = MultiModalFieldElem(
        data=None,
        field=MultiModalSharedField(batch_size=2),
    )
    item = MultiModalKwargsItem({"empty": elem})
    encoded = encode_mm_kwargs_item(item)
    decoded = decode_mm_kwargs_item(encoded)

    assert decoded["empty"].data is None
    assert isinstance(decoded["empty"].field, MultiModalSharedField)


def test_mm_kwargs_item_nested_tensors():
    """Roundtrip with nested tensor data."""
    nested = [torch.randn(3, 4), torch.randn(5, 4)]
    elem = MultiModalFieldElem(
        data=nested,
        field=MultiModalBatchedField(),
    )
    item = MultiModalKwargsItem({"nested": elem})
    encoded = encode_mm_kwargs_item(item)
    decoded = decode_mm_kwargs_item(encoded)

    decoded_data = decoded["nested"].data
    assert len(decoded_data) == 2
    assert torch.equal(nested[0], decoded_data[0])
    assert torch.equal(nested[1], decoded_data[1])


def test_mm_features_with_kwargs_data():
    """Test that MultiModalFeatures can carry serialized tensor data."""
    elem = MultiModalFieldElem(
        data=torch.randn(5, 3, dtype=torch.float32),
        field=MultiModalBatchedField(),
    )
    item = MultiModalKwargsItem({"pixel_values": elem})
    encoded = encode_mm_kwargs_item(item)

    features = MultiModalFeatures(
        mm_hashes={"image": ["abc123"]},
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=0, length=10)]},
        kwargs_data={"image": [encoded]},
    )

    # JSON roundtrip
    json_str = features.model_dump_json()
    features2 = MultiModalFeatures.model_validate_json(json_str)

    assert features2.mm_hashes == {"image": ["abc123"]}
    assert features2.kwargs_data is not None
    assert len(features2.kwargs_data["image"]) == 1

    decoded = decode_mm_kwargs_item(features2.kwargs_data["image"][0])
    assert torch.equal(elem.data, decoded["pixel_values"].data)
