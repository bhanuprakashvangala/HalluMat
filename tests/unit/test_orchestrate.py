"""The unattended orchestrator against a fake cluster."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DEPLOY = Path(__file__).parents[2] / "deploy" / "nautilus"


@pytest.fixture(scope="module")
def orchestrate():
    sys.path.insert(0, str(DEPLOY))
    spec = importlib.util.spec_from_file_location("orchestrate", DEPLOY / "orchestrate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["orchestrate"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("orchestrate", None)
    sys.path.remove(str(DEPLOY))


def job(name, retry, condition=None, active=0):
    conditions = [SimpleNamespace(type=condition, status="True")] if condition else []
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, annotations={"hallumat/retry": str(retry)}),
        status=SimpleNamespace(conditions=conditions, active=active),
    )


class FakeCluster:
    """Jobs finish in the order scripted per unit: each create appends the next scripted outcome."""

    def __init__(self, outcomes):
        self.outcomes, self.jobs, self.created = outcomes, {}, []

    def jobs_of(self, unit):
        return self.jobs.get(unit, [])

    def create(self, manifest):
        unit = manifest["metadata"]["labels"]["hallumat-unit"]
        retry = int(manifest["metadata"]["annotations"]["hallumat/retry"])
        self.created.append(manifest["metadata"]["name"])
        outcome = self.outcomes[unit].pop(0)
        self.jobs.setdefault(unit, []).append(job(manifest["metadata"]["name"], retry, outcome))


def test_plan_covers_every_stage_in_order(orchestrate):
    steps = orchestrate.plan()
    assert [units[0].stage for _, units in steps] == [
        "wiki-questions", "select", "paraphrase", "respond", "label", "export", "publish",
    ]  # fmt: skip
    assert len(steps[3][1]) == 8
    assert all(u.engine == "api" for u in steps[0][1] + steps[2][1])


def test_unit_manifest_is_labelled_and_retried_under_a_new_name(orchestrate):
    unit = orchestrate.Unit("respond", model="qwen3-8b", shard="1/2")
    first, second = unit.manifest("abc", 0), unit.manifest("abc", 2)
    assert first["metadata"]["name"] == "bhanu-hallumat-respond-qwen3-8b-1of2"
    assert second["metadata"]["name"] == "bhanu-hallumat-respond-qwen3-8b-1of2-r2"
    assert first["metadata"]["labels"]["hallumat-unit"] == second["metadata"]["labels"]["hallumat-unit"]
    assert second["spec"]["template"]["metadata"]["labels"]["hallumat-unit"] == first["metadata"]["name"]


def test_failed_jobs_are_retried_until_success(orchestrate, monkeypatch):
    monkeypatch.setattr(orchestrate.time, "sleep", lambda s: None)
    unit = orchestrate.Unit("label")
    name = unit.manifest("abc", 0)["metadata"]["labels"]["hallumat-unit"]
    cluster = FakeCluster({name: ["Failed", "Failed", "Complete"]})
    reports = []
    orchestrate.run_step(cluster, "abc", [unit], reports.append)
    assert cluster.created == [name, f"{name}-r1", f"{name}-r2"]
    assert reports[-1] == {name: "succeeded"}


def test_a_unit_that_keeps_failing_stops_the_run(orchestrate, monkeypatch):
    monkeypatch.setattr(orchestrate.time, "sleep", lambda s: None)
    unit = orchestrate.Unit("export")
    name = unit.manifest("abc", 0)["metadata"]["labels"]["hallumat-unit"]
    cluster = FakeCluster({name: ["Failed"] * (orchestrate.MAX_RETRIES + 1)})
    with pytest.raises(RuntimeError, match="failed 5 times"):
        orchestrate.run_step(cluster, "abc", [unit], lambda states: None)


def test_existing_successful_jobs_are_not_recreated(orchestrate):
    unit = orchestrate.Unit("select")
    name = unit.manifest("abc", 0)["metadata"]["labels"]["hallumat-unit"]
    cluster = FakeCluster({})
    cluster.jobs[name] = [job(name, 0, "Complete")]
    orchestrate.run_step(cluster, "abc", [unit], lambda states: None)
    assert cluster.created == []
