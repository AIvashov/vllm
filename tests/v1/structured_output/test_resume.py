# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.request import StructuredOutputRequest


def test_structured_output_request_reads_resume_token_ids():
    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json='{"type": "object"}'),
        extra_args={"structured_output_resume_token_ids": [1, 2, 3]},
    )

    request = StructuredOutputRequest.from_sampling_params(sampling_params)

    assert request is not None
    assert request.resume_token_ids == [1, 2, 3]
    assert (
        request.resume_token_ids
        is not sampling_params.extra_args["structured_output_resume_token_ids"]
    )


@pytest.mark.parametrize(
    "resume_token_ids",
    [
        "1,2,3",
        [1, "2", 3],
        [1, True, 3],
        [1, -1, 3],
    ],
)
def test_structured_output_request_rejects_invalid_resume_token_ids(
    resume_token_ids,
):
    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(json='{"type": "object"}'),
        extra_args={"structured_output_resume_token_ids": resume_token_ids},
    )

    with pytest.raises(ValueError, match="structured_output_resume_token_ids"):
        StructuredOutputRequest.from_sampling_params(sampling_params)


def test_create_grammar_replays_resume_token_ids():
    manager = object.__new__(StructuredOutputManager)
    grammar = Mock()
    grammar.accept_tokens.return_value = True
    manager.backend = Mock()
    manager.backend.compile_grammar.return_value = grammar

    structured_output_request = StructuredOutputRequest(
        params=StructuredOutputsParams(json='{"type": "object"}'),
        resume_token_ids=[10, 11],
    )
    request = Mock(
        request_id="resume-request",
        structured_output_request=structured_output_request,
    )

    result = manager._create_grammar(request)

    assert result is grammar
    manager.backend.compile_grammar.assert_called_once_with(
        StructuredOutputOptions.JSON, '{"type": "object"}'
    )
    grammar.accept_tokens.assert_called_once_with("resume-request", [10, 11])


def test_create_grammar_rejects_invalid_resume_prefix():
    manager = object.__new__(StructuredOutputManager)
    grammar = Mock()
    grammar.accept_tokens.return_value = False
    manager.backend = Mock()
    manager.backend.compile_grammar.return_value = grammar

    request = Mock(
        request_id="resume-request",
        structured_output_request=StructuredOutputRequest(
            params=StructuredOutputsParams(json='{"type": "object"}'),
            resume_token_ids=[10, 11],
        ),
    )

    with pytest.raises(ValueError, match="rejected resume prefix"):
        manager._create_grammar(request)

    grammar.accept_tokens.assert_called_once_with("resume-request", [10, 11])
