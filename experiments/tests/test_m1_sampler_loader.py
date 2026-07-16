from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from experiments.m1 import workflow
from experiments.m1.contracts import M1ConfigError


BASE_MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def _sampler_config(manifest_path: Path) -> dict[str, object]:
    return {
        "model": {
            "base": BASE_MODEL,
            "revision": REVISION,
        },
        "sampling": {
            "checkpoints_manifest": str(manifest_path),
        },
    }


def _write_manifest(path: Path, *, model_revision: str = REVISION) -> None:
    payload = {
        "protocol_version": "m1.checkpoints.v1",
        "model_base": BASE_MODEL,
        "model_revision": model_revision,
        "checkpoints": [
            {
                "seed": 11,
                "checkpoint_dir": "/checkpoints/runs/example/seed-11",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _install_fake_generation_modules(monkeypatch: pytest.MonkeyPatch, *, adapter_base: str):
    calls: dict[str, list[tuple[object, ...]]] = {
        "models": [],
        "tokenizers": [],
        "peft": [],
    }

    torch_module = ModuleType("torch")
    torch_module.manual_seed = lambda seed: None
    torch_module.use_deterministic_algorithms = lambda enabled: None
    torch_module.cuda = SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda seed: None,
    )

    class _InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_module.inference_mode = _InferenceMode
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    peft_module = ModuleType("peft")

    class FakePeftConfig:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls["peft"].append(("config", source, kwargs))
            return SimpleNamespace(base_model_name_or_path=adapter_base)

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, source, **kwargs):
            calls["peft"].append(("model", source, kwargs))
            return model

    peft_module.PeftConfig = FakePeftConfig
    peft_module.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    transformers_module = ModuleType("transformers")

    class FakeModel:
        def eval(self):
            return None

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls["models"].append((source, kwargs))
            return FakeModel()

    class FakeTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"
        padding_side = "right"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls["tokenizers"].append((source, kwargs))
            return FakeTokenizer()

    transformers_module.AutoModelForCausalLM = FakeAutoModelForCausalLM
    transformers_module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    return calls


def test_load_checkpoint_index_rejects_manifest_revision_mismatch(tmp_path: Path) -> None:
    manifest_path = tmp_path / "checkpoints_manifest.json"
    _write_manifest(manifest_path, model_revision="different-revision")

    with pytest.raises(M1ConfigError, match="checkpoint manifest model_revision mismatch"):
        workflow._load_checkpoint_index(_sampler_config(manifest_path))


def test_generate_outputs_loads_peft_base_and_tokenizer_with_exact_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "seed-11"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    calls = _install_fake_generation_modules(monkeypatch, adapter_base=BASE_MODEL)

    outputs = workflow._generate_outputs(
        checkpoint_dir=checkpoint_dir,
        base_model=BASE_MODEL,
        base_revision=REVISION,
        records=[],
        prompt_format="USER: {user_prompt}\nASSISTANT:",
        max_input_tokens=1024,
        max_new_tokens=384,
        temperature=1.0,
        top_p=1.0,
        sampling_seed=101,
        do_sample=True,
    )

    assert outputs == []
    assert calls["models"] == [
        (
            BASE_MODEL,
            {
                "revision": REVISION,
                "local_files_only": True,
                "trust_remote_code": True,
            },
        )
    ]
    assert calls["tokenizers"] == [
        (
            BASE_MODEL,
            {
                "local_files_only": True,
                "trust_remote_code": True,
                "revision": REVISION,
            },
        )
    ]


def test_generate_outputs_rejects_adapter_base_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "seed-11"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    _install_fake_generation_modules(monkeypatch, adapter_base="Qwen/Qwen3-0.6B")

    with pytest.raises(M1ConfigError, match="adapter base model mismatch"):
        workflow._generate_outputs(
            checkpoint_dir=checkpoint_dir,
            base_model=BASE_MODEL,
            base_revision=REVISION,
            records=[],
            prompt_format="USER: {user_prompt}\nASSISTANT:",
            max_input_tokens=1024,
            max_new_tokens=384,
            temperature=1.0,
            top_p=1.0,
            sampling_seed=101,
            do_sample=True,
        )
