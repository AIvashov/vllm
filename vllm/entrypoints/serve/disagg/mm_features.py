# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.entrypoints.serve.disagg.mm_serde import decode_mm_kwargs_item
from vllm.entrypoints.serve.disagg.protocol import MultiModalFeatures
from vllm.inputs import MultiModalInput, mm_input
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
)


def restore_mm_placeholders(
    features: MultiModalFeatures,
) -> dict[str, list[PlaceholderRange]]:
    """Convert serialized placeholder metadata back to engine ranges."""
    return {
        modality: [
            PlaceholderRange(
                offset=p.offset,
                length=p.length,
                is_embed=(
                    None
                    if p.is_embed is None
                    else torch.tensor(p.is_embed, dtype=torch.bool)
                ),
            )
            for p in ranges
        ]
        for modality, ranges in features.mm_placeholders.items()
    }


def restore_mm_kwargs(
    features: MultiModalFeatures,
) -> dict[str, list[MultiModalKwargsItem | None]]:
    """Deserialize multimodal kwargs, using None entries for cache hits."""
    if features.kwargs_data is not None:
        return {
            modality: [
                decode_mm_kwargs_item(item) if item is not None else None
                for item in items
            ]
            for modality, items in features.kwargs_data.items()
        }

    return {
        modality: [None] * len(hashes)
        for modality, hashes in features.mm_hashes.items()
    }


def build_mm_input_from_features(
    *,
    token_ids: list[int],
    features: MultiModalFeatures,
    cache_salt: str | None = None,
) -> MultiModalInput:
    return mm_input(
        prompt_token_ids=token_ids,
        mm_kwargs=MultiModalKwargsItems(restore_mm_kwargs(features)),
        mm_hashes=features.mm_hashes,
        mm_placeholders=restore_mm_placeholders(features),
        cache_salt=cache_salt,
    )
