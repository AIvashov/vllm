# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.logprobs import FlatLogprobs, create_prompt_logprobs
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.outputs import LogprobsTensors


class _FakeTokenizer:
    def __init__(self):
        self.decoded_token_ids: list[int] = []

    def decode(self, token_ids: list[int]) -> str:
        self.decoded_token_ids.extend(token_ids)
        if any(token_id < 0 for token_id in token_ids):
            raise OverflowError("negative token id passed to decode")
        return f"tok{token_ids[0]}"


def test_empty_logprobs_tensors_use_missing_sentinels():
    tensors = LogprobsTensors.empty_cpu(num_positions=2, num_tokens_per_position=3)

    assert torch.all(tensors.logprob_token_ids == -1)
    assert torch.isnan(tensors.logprobs).all()
    assert torch.all(tensors.selected_token_ranks == -1)


@pytest.mark.parametrize("flat_logprobs", [False, True])
def test_prompt_logprobs_skip_missing_positions(flat_logprobs: bool):
    tokenizer = _FakeTokenizer()
    processor = LogprobsProcessor(
        tokenizer=tokenizer,
        logprobs=None,
        prompt_logprobs=create_prompt_logprobs(flat_logprobs),
        cumulative_logprob=None,
        num_logprobs=None,
        num_prompt_logprobs=1,
    )

    prompt_logprobs_tensors = LogprobsTensors(
        logprob_token_ids=torch.tensor(
            [
                [-1, -1],
                [42, 7],
                [-1, -1],
                [99, 8],
            ],
            dtype=torch.int32,
        ),
        logprobs=torch.tensor(
            [
                [math.nan, math.nan],
                [-0.1, -1.0],
                [math.nan, math.nan],
                [-0.2, -2.0],
            ],
            dtype=torch.float32,
        ),
        selected_token_ranks=torch.tensor([-1, 1, -1, 2], dtype=torch.int32),
    )

    processor._update_prompt_logprobs(prompt_logprobs_tensors)

    prompt_logprobs = processor.prompt_logprobs
    assert prompt_logprobs is not None
    assert len(prompt_logprobs) == 5
    assert tokenizer.decoded_token_ids == [42, 7, 99, 8]

    if isinstance(prompt_logprobs, FlatLogprobs):
        assert not prompt_logprobs[0]
        assert not prompt_logprobs[1]
        assert not prompt_logprobs[3]
    else:
        assert prompt_logprobs[0] is None
        assert prompt_logprobs[1] is None
        assert prompt_logprobs[3] is None

    assert prompt_logprobs[2][42].logprob == pytest.approx(-0.1)
    assert prompt_logprobs[2][42].decoded_token == "tok42"
    assert prompt_logprobs[4][99].logprob == pytest.approx(-0.2)
    assert prompt_logprobs[4][99].decoded_token == "tok99"
    assert LogprobsProcessor._get_sampled_context_ids(prompt_logprobs) == [42, 99]
