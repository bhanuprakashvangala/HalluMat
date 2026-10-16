"""Batch chat generation. `VLLMBackend` runs on a GPU with vLLM; `TransformersBackend` is a slow fallback for small
local runs and tests. Both take a list of conversations and return one reply per conversation."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from hallumat._optional import require
from hallumat.errors import HallumatError

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

    def __init__(
        self,
        model: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        vllm = require("vllm", "vllm", "extend")
        self.name = model
        self._vllm = vllm
        self._template_kwargs = chat_template_kwargs or {}
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
        extra = {"chat_template_kwargs": self._template_kwargs} if self._template_kwargs else {}
        outputs = self._llm.chat([list(c) for c in conversations], params, use_tqdm=False, **extra)
        return [strip_reasoning(output.outputs[0].text) for output in outputs]


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


class APIBackend:
    """An OpenAI-compatible chat endpoint (by default the NRP managed LLMs), several requests at a time.

    The key comes from $NRP_KEY or ~/.nrp_key, the endpoint from $NRP_BASE. Requests are retried with growing
    pauses, as the NRP fair-use policy asks: on HTTP 429 the pause follows the x-ratelimit-reset header, and a
    request that keeps failing is given up after about two hours (its record is then simply missing and the stage
    can be rerun).
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        concurrency: int = 12,
        timeout: float = 600,
        chat_template_kwargs: dict[str, Any] | None = None,
        session: Any = None,
    ) -> None:
        self.name = model
        self.base_url = (base_url or os.environ.get("NRP_BASE") or NRP_BASE).rstrip("/")
        self.api_key = api_key or nrp_key()
        self.concurrency = concurrency
        self.timeout = timeout
        self._template_kwargs = chat_template_kwargs or {}
        self._session = session or require("requests", "requests", "detector").Session()

    def _one(self, messages: Sequence[Message], sampling: Sampling, seed: int) -> str:
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": list(messages),
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "seed": seed,
        }
        if self._template_kwargs:
            payload["chat_template_kwargs"] = self._template_kwargs
        headers = {"Authorization": f"Bearer {self.api_key}"}
        waited = 0.0
        for attempt in range(1000):
            pause = min(300.0, 5.0 * 2**attempt)
            try:
                response = self._session.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout
                )
                if response.status_code == 200:
                    message = response.json()["choices"][0]["message"]
                    return strip_reasoning(message.get("content") or "")
                if response.status_code == 429:
                    pause = float(response.headers.get("x-ratelimit-reset", 30)) + 1
                elif response.status_code in (400, 401, 403, 404):
                    raise RuntimeError(f"{self.name}: HTTP {response.status_code}: {response.text[:300]}")
                log.warning("%s: HTTP %s, retrying in %.0f s", self.name, response.status_code, pause)
            except RuntimeError:
                raise
            except Exception as exc:
                log.warning("%s: %s, retrying in %.0f s", self.name, exc, pause)
            if waited > 7200:
                log.error("%s: giving up on one request after %.0f s", self.name, waited)
                return ""
            time.sleep(pause)
            waited += pause
        return ""

    def chat(
        self, conversations: Sequence[Sequence[Message]], sampling: Sampling, seeds: Sequence[int] | None = None
    ) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        resolved = resolve_seeds(len(conversations), sampling, seeds)
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(
                pool.map(lambda pair: self._one(pair[0], sampling, pair[1]), zip(conversations, resolved, strict=True))
            )


NRP_BASE = "https://ellm.nrp-nautilus.io/v1"


def nrp_key() -> str:
    """The NRP LLM token from $NRP_KEY or ~/.nrp_key."""
    key = os.environ.get("NRP_KEY", "").strip()
    path = os.path.expanduser("~/.nrp_key")
    if not key and os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            key = handle.read().strip()
    if not key:
        raise HallumatError("no NRP LLM token: set $NRP_KEY or write it to ~/.nrp_key")
    return key


_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """The reply without a <think>...</think> reasoning block, if a model emitted one."""
    return _THINK.sub("", text).strip()


def make_backend(model: str, engine: str = "vllm", **kwargs: Any) -> ChatBackend:
    if engine == "vllm":
        return VLLMBackend(model, **kwargs)
    if engine == "api":
        allowed = {"concurrency", "chat_template_kwargs", "base_url"}
        return APIBackend(model, **{k: v for k, v in kwargs.items() if k in allowed})
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
