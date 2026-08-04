import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import local_qwen


class FakeInputs(dict):
    def __init__(self):
        input_ids = torch.tensor([[1, 2]])
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, _device):
        return self


class FakeProcessor:
    def apply_chat_template(self, _messages, **_kwargs):
        return "prompt"

    def __call__(self, **_kwargs):
        return FakeInputs()

    def batch_decode(self, _tokens, **_kwargs):
        return ["generated"]


class FakeModel:
    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3]])


def generation_context(model, processor):
    qwen_utils = types.SimpleNamespace(
        process_vision_info=lambda _messages: (None, None)
    )
    return (
        mock.patch.dict(sys.modules, {"qwen_vl_utils": qwen_utils}),
        mock.patch.object(local_qwen, "_load_qwen", return_value=(model, processor)),
    )


def test_schema_generation_installs_token_constraint_and_explicit_settings():
    model = FakeModel()
    processor = FakeProcessor()
    constraint = mock.Mock(name="prefix_constraint")
    modules, loader = generation_context(model, processor)
    with modules, loader:
        with mock.patch.object(
            local_qwen,
            "_json_schema_prefix_allowed_tokens_fn",
            return_value=constraint,
        ) as build_constraint:
            result = local_qwen.qwen_generate(
                [{"role": "user", "content": "interpret"}],
                model_id="fake",
                local_files_only=True,
                device_map="cpu",
                do_sample=False,
                repetition_penalty=1.0,
                json_schema={"type": "object"},
            )
    assert result == "generated"
    assert model.generate_kwargs["prefix_allowed_tokens_fn"] is constraint
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["repetition_penalty"] == 1.0
    build_constraint.assert_called_once()


def test_preload_warms_constraint_tokenizer_metadata_before_health_ready():
    model = FakeModel()
    model.hf_device_map = {"": "cuda:0"}
    model.parameters = lambda: iter([torch.zeros(1, dtype=torch.bfloat16)])
    processor = FakeProcessor()
    constraint_data = types.SimpleNamespace(vocab_size=151665)
    with mock.patch.object(
        local_qwen,
        "_load_qwen",
        return_value=(model, processor),
    ):
        with mock.patch.object(
            local_qwen,
            "_lmfe_tokenizer_data",
            return_value=constraint_data,
        ) as warm_constraint:
            metadata = local_qwen.preload_qwen(
                "fake",
                local_files_only=True,
                device_map="auto",
            )
    assert metadata["json_schema_constraint_ready"] is True
    assert metadata["constraint_vocab_size"] == 151665
    warm_constraint.assert_called_once_with("fake", True, "auto")


def test_schema_generation_rejects_response_prefix_before_model_loading():
    with pytest.raises(ValueError, match="cannot be combined"):
        local_qwen.qwen_generate(
            [{"role": "user", "content": "interpret"}],
            response_prefix="{",
            json_schema={"type": "object"},
        )


def test_schema_constraint_failure_is_fail_closed():
    model = FakeModel()
    processor = FakeProcessor()
    modules, loader = generation_context(model, processor)
    with modules, loader:
        with mock.patch.object(
            local_qwen,
            "_json_schema_prefix_allowed_tokens_fn",
            side_effect=RuntimeError("constraint unavailable"),
        ):
            with pytest.raises(RuntimeError, match="constraint unavailable"):
                local_qwen.qwen_generate(
                    [{"role": "user", "content": "interpret"}],
                    model_id="fake",
                    local_files_only=True,
                    device_map="cpu",
                    json_schema={"type": "object"},
                )
    assert model.generate_kwargs is None


def test_unconstrained_visual_calls_remain_unchanged():
    model = FakeModel()
    processor = FakeProcessor()
    modules, loader = generation_context(model, processor)
    with modules, loader:
        with mock.patch.object(
            local_qwen,
            "_json_schema_prefix_allowed_tokens_fn",
        ) as build_constraint:
            result = local_qwen.qwen_generate(
                [{"role": "user", "content": "verify"}],
                model_id="fake",
                local_files_only=True,
                device_map="cpu",
                response_prefix='{"decision":',
            )
    assert result == '{"decision":generated'
    assert "prefix_allowed_tokens_fn" not in model.generate_kwargs
    build_constraint.assert_not_called()
