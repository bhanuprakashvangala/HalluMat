"""Run the whole dataset extension on Nautilus, unattended.

Runs inside the cluster (see orchestrator.yaml): creates the Job of every stage in order with the Kubernetes API,
waits for all Jobs of a step to succeed, recreates a failed Job up to MAX_RETRIES times (each stage resumes from
the files already written), and records progress in /work/status.json. Restarting the orchestrator is safe: it
finds the Jobs it created earlier by their labels and carries on.

    python deploy/nautilus/orchestrate.py --ref <commit>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import job as jobs

log = logging.getLogger("orchestrate")

MAX_RETRIES = 4
POLL_SECONDS = 60
RESPONDERS = ("llama2-7b-chat", "llama3.1-8b", "qwen3-8b", "gemma3-12b")


@dataclass
class Unit:
    """One Job of a step: a stage, optionally for one model and one shard."""

    stage: str
    model: str | None = None
    shard: str | None = None
    engine: str = "vllm"
    concurrency: int = 8
    passthrough: tuple[str, ...] = ()

    def manifest(self, ref: str, retry: int) -> dict:
        args = SimpleNamespace(
            stage=self.stage,
            ref=ref,
            model=self.model,
            shard=self.shard,
            tensor_parallel=2,
            writer=None,
            gpu_products=None,
            engine=self.engine,
            concurrency=self.concurrency,
            name_suffix=f"-r{retry}" if retry else "",
            passthrough=list(self.passthrough),
        )
        manifest = jobs.render(args)
        base = jobs.render(SimpleNamespace(**{**vars(args), "name_suffix": ""}))["metadata"]["name"]
        for labels in (manifest["metadata"]["labels"], manifest["spec"]["template"]["metadata"]["labels"]):
            labels["hallumat-unit"] = base
        manifest["metadata"].setdefault("annotations", {})["hallumat/retry"] = str(retry)
        return manifest


def plan(writer_shards: int = 2, respond_shards: int = 2) -> list[tuple[str, list[Unit]]]:
    shards = lambda n: [f"{i}/{n}" for i in range(n)]  # noqa: E731
    return [
        ("write questions", [Unit("wiki-questions", shard=s, engine="api") for s in shards(writer_shards)]),
        ("select 50,000 questions", [Unit("select", passthrough=("--total", "50000"))]),
        ("paraphrase", [Unit("paraphrase", shard=s, engine="api") for s in shards(writer_shards)]),
        ("answer", [Unit("respond", model=m, shard=s) for m in RESPONDERS for s in shards(respond_shards)]),
        ("label", [Unit("label")]),
        ("export", [Unit("export")]),
        ("publish (Hugging Face pull request)", [Unit("publish")]),
    ]


class Cluster:
    def __init__(self, namespace: str) -> None:
        from kubernetes import client, config

        config.load_incluster_config()
        self.batch = client.BatchV1Api()
        self.namespace = namespace

    def jobs_of(self, unit_name: str) -> list:
        items = self.batch.list_namespaced_job(self.namespace, label_selector=f"hallumat-unit={unit_name}").items
        return sorted(items, key=lambda j: int((j.metadata.annotations or {}).get("hallumat/retry", 0)))

    def create(self, manifest: dict) -> None:
        self.batch.create_namespaced_job(self.namespace, manifest)


def state(job) -> str:
    status = job.status
    for condition in status.conditions or []:
        if condition.status == "True" and condition.type in ("Complete", "Failed"):
            return "succeeded" if condition.type == "Complete" else "failed"
    return "running" if (status.active or 0) else "pending"


def run_step(cluster: Cluster, ref: str, units: list[Unit], report) -> None:
    while True:
        states = {}
        for unit in units:
            base = unit.manifest(ref, 0)["metadata"]["labels"]["hallumat-unit"]
            existing = cluster.jobs_of(base)
            latest = existing[-1] if existing else None
            current = state(latest) if latest else "absent"
            retry = int((latest.metadata.annotations or {}).get("hallumat/retry", 0)) if latest else -1
            if current == "absent" or (current == "failed" and retry < MAX_RETRIES):
                manifest = unit.manifest(ref, retry + 1 if latest else 0)
                log.info("creating %s", manifest["metadata"]["name"])
                cluster.create(manifest)
                current = "created"
            elif current == "failed":
                raise RuntimeError(f"{base} failed {MAX_RETRIES + 1} times; see its logs")
            states[base] = current
        report(states)
        if all(s == "succeeded" for s in states.values()):
            return
        time.sleep(POLL_SECONDS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--status", type=Path, default=Path("/work/status.json"))
    parser.add_argument("--writer-shards", type=int, default=2)
    parser.add_argument("--respond-shards", type=int, default=2)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cluster = Cluster(jobs.NAMESPACE)
    steps = plan(args.writer_shards, args.respond_shards)
    status: dict = {"ref": args.ref, "started": dt.datetime.now(dt.UTC).isoformat(), "steps": {}}

    for index, (title, units) in enumerate(steps, 1):

        def report(states: dict, title: str = title, index: int = index) -> None:
            status["current"] = f"{index}/{len(steps)} {title}"
            status["steps"][title] = states
            status["updated"] = dt.datetime.now(dt.UTC).isoformat()
            args.status.write_text(json.dumps(status, indent=2), encoding="utf-8")

        log.info("step %d/%d: %s (%d jobs)", index, len(steps), title, len(units))
        run_step(cluster, args.ref, units, report)
        log.info("step %d/%d done: %s", index, len(steps), title)
    status["current"] = "finished"
    status["finished"] = dt.datetime.now(dt.UTC).isoformat()
    args.status.write_text(json.dumps(status, indent=2), encoding="utf-8")
    log.info("all steps finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
