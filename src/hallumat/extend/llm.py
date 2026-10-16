"""Batch chat generation. `VLLMBackend` runs on a GPU with vLLM; `TransformersBackend` is a slow fallback for small
local runs and tests. Both take a list of conversations and return one reply per conversation."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from hallumat._optional import require

log = logging.getLogger(__name__)

Message = dict[str, str]


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    seed: int = 0


class ChatBackend(Protocol):
    name: str

    def chat(
        self, conversations: Sequence[Sequence[Message]], sampling: Sampling, seeds: Sequence[int] | None = None
    ) -> list[str]: ...


def resolve_seeds(n: int, sampling: Sampling, seeds: Sequence[int] | None) -> list[int]:
    """One seed per conversation: the given ones, or sampling.seed + index."""
    if seeds is not None:
        if len(seeds) != n:
            raise ValueError(f"{len(seeds)} seeds for {n} conversations")
        return list(seeds)
    return [sampling.seed + i for i in range(n)]


class VLLMBackend:
    """vLLM offline inference; one engine per process. Each conversation is sampled with its own seed, so a rerun
    samples the same text."""

    def __init__(self, model: str, max_model_len: int = 4096, gpu_memory_utilization: float = 0.9, **kwargs: Any):
        vllm = require("vllm", "vllm", "extend")
        self.name = model
        self._vllm = vllm
        self._llm = vllm.LLM(
            model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, **kwargs
        )

    def chat(
        self, conversations: Sequence[Sequence[Message]], sampling: Sampling, seeds: Sequence[int] | None = None
    ) -> list[str]:
        params = [
            self._vllm.SamplingParams(
                temperature=sampling.temperature, top_p=sampling.top_p, max_tokens=sampling.max_tokens, seed=seed
            )
            for seed in resolve_seeds(len(conversations), sampling, seeds)
        ]
        outputs = self._llm.chat([list(c) for c in conversations], params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]


class TransformersBackend:
    """transformers generate(), one conversation at a time."""

    def __init__(self, model: str) -> None:
        transformers = require("transformers", "transformers", "models")
        self._torch: Any = require("torch", "torch", "models")
        self.name = model
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        self._model = transformers.AutoModelForCausalLM.from_pretrained(model, device_map="auto")

    def chat(
        self, conversations: Sequence[Sequence[Message]], sampling: Sampling, seeds: Sequence[int] | None = None
    ) -> list[str]:
        replies = []
        for seed, conversation in zip(resolve_seeds(len(conversations), sampling, seeds), conversations, strict=True):
            self._torch.manual_seed(seed)
            ids = self._tokenizer.apply_chat_template(
                list(conversation), add_generation_prompt=True, return_tensors="pt"
            ).to(self._model.device)
            out = self._model.generate(
                ids,
                max_new_tokens=sampling.max_tokens,
                do_sample=sampling.temperature > 0,
                temperature=sampling.temperature or None,
                top_p=sampling.top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            replies.append(str(self._tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)))
        return replies


def make_backend(model: str, engine: str = "vllm", **kwargs: Any) -> ChatBackend:
    if engine == "vllm":
        return VLLMBackend(model, **kwargs)
    return TransformersBackend(model)


_JSON_BLOCK = re.compile(r"(\[.*\]|\{.*\})", re.DOTALL)


def parse_json(reply: str) -> Any:
    """The first JSON array or object in a model reply (code fences and prose around it are ignored), or None."""
    reply = reply.strip()
    if reply.startswith("```"):
        reply = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", reply)
    match = _JSON_BLOCK.search(reply)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
