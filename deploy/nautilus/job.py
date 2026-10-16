"""Print the Kubernetes Job that runs one `hallumat extend` stage on NRP Nautilus.

    python deploy/nautilus/job.py respond --model llama3.1-8b --shard 0/2 --ref <commit> | kubectl apply -f -

Jobs install hallumat from GitHub at --ref, read and write the work volume (bhanu-hallumat-work, mounted at /work)
and keep model weights in the shared Hugging Face cache (bhanu-pickspin-hf-cache, mounted at /cache). GPU stages run
in the vLLM image; limits equal requests, and every Job has a deadline, as NRP requires.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

NAMESPACE = "gp-engine-mizzou-radiant"
REPO = "https://github.com/bhanuprakashvangala/HalluMat"
VLLM_IMAGE = "vllm/vllm-openai:v0.8.5"
CPU_IMAGE = "python:3.12-slim"
GPU_48GB = ["NVIDIA-L40", "NVIDIA-L40S", "NVIDIA-RTX-A6000", "NVIDIA-A40"]
GPU_24GB = ["NVIDIA-GeForce-RTX-3090", "NVIDIA-A10", "NVIDIA-GeForce-RTX-4090", "NVIDIA-L4", *GPU_48GB]

# stage: (image, gpus, gpu products, cpu, memory, extra pip packages, deadline hours)
STAGES = {
    "fetch-mp": (CPU_IMAGE, 0, [], 2, "8Gi", "requests pyarrow", 2),
    "fetch-wiki": (CPU_IMAGE, 0, [], 2, "8Gi", "requests pyarrow", 8),
    "mp-questions": (CPU_IMAGE, 0, [], 2, "8Gi", "requests pyarrow", 1),
    "wiki-questions": (VLLM_IMAGE, 2, GPU_48GB, 8, "64Gi", "", 12),
    "select": (VLLM_IMAGE, 0, [], 8, "16Gi", "sentence-transformers faiss-cpu", 3),
    "paraphrase": (VLLM_IMAGE, 2, GPU_48GB, 8, "64Gi", "", 12),
    "respond": (VLLM_IMAGE, 1, GPU_24GB, 6, "40Gi", "", 12),
    "label": (VLLM_IMAGE, 1, GPU_24GB, 6, "32Gi", "", 8),
    "export": (CPU_IMAGE, 0, [], 4, "32Gi", "pyarrow huggingface_hub", 3),
}
# Gemma-3-12B needs 24 GB of weights in bf16, so a 48 GB card.
MODEL_GPUS = {"gemma3-12b": GPU_48GB}


def render(args: argparse.Namespace) -> dict[str, object]:
    image, gpus, products, cpu, memory, extra, hours = STAGES[args.stage]
    products = MODEL_GPUS.get(args.model or "", products)
    if args.stage in ("wiki-questions", "paraphrase"):
        gpus = args.tensor_parallel
    hallumat_args = ["--work", "/work", *args.passthrough]
    if args.model:
        hallumat_args += ["--model", args.model]
    if args.shard:
        hallumat_args += ["--shard", args.shard]
    if gpus > 1:
        hallumat_args += ["--tensor-parallel", str(gpus)]
    suffix = "-".join(x for x in (args.model, args.shard.replace("/", "of") if args.shard else None) if x)
    name = re.sub(r"[^a-z0-9-]", "-", f"bhanu-hallumat-{args.stage}{'-' + suffix if suffix else ''}{args.name_suffix}")

    # vLLM's image already has torch and transformers: install hallumat without dependencies there, so pip never
    # replaces them, plus the few light packages the stages import.
    if image == VLLM_IMAGE:
        install = (
            "pip freeze | grep -iE '^(numpy|torch|transformers|tokenizers)==' > /tmp/pins.txt\n"
            f"pip install -q --no-deps 'hallumat @ git+{REPO}@{args.ref}'\n"
            f"pip install -q -c /tmp/pins.txt pandas tabulate scikit-learn seaborn pyarrow requests {extra}"
        )
    else:
        install = (
            "apt-get update -qq && apt-get install -y -qq git >/dev/null\n"
            f"pip install -q 'hallumat @ git+{REPO}@{args.ref}' {extra}"
        )
    script = f"set -e\n{install}\nhallumat -v extend {args.stage} {' '.join(hallumat_args)}"

    env = [
        {"name": "HF_HOME", "value": "/cache"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": "bhanu-pickspin-hf", "key": "HF_TOKEN"}}},
    ]
    if args.stage == "fetch-mp":
        env.append(
            {"name": "MP_API_KEY", "valueFrom": {"secretKeyRef": {"name": "bhanu-hallumat-mp", "key": "MP_API_KEY"}}}
        )
    resources: dict[str, object] = {"cpu": str(cpu), "memory": memory}
    if gpus:
        resources["nvidia.com/gpu"] = str(gpus)
    pod: dict[str, object] = {
        "restartPolicy": "Never",
        "containers": [
            {
                "name": "stage",
                "image": image,
                "command": ["bash", "-c", script],
                "env": env,
                "resources": {"requests": resources, "limits": resources},
                "volumeMounts": [
                    {"name": "work", "mountPath": "/work"},
                    {"name": "hf-cache", "mountPath": "/cache"},
                    {"name": "dshm", "mountPath": "/dev/shm"},
                ],
            }
        ],
        "volumes": [
            {"name": "work", "persistentVolumeClaim": {"claimName": "bhanu-hallumat-work"}},
            {"name": "hf-cache", "persistentVolumeClaim": {"claimName": "bhanu-pickspin-hf-cache"}},
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
        ],
    }
    if products:
        pod["affinity"] = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {"matchExpressions": [{"key": "nvidia.com/gpu.product", "operator": "In", "values": products}]}
                    ]
                }
            }
        }
    else:
        pod["nodeSelector"] = {"kubernetes.io/arch": "amd64"}
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": name[:63].rstrip("-"), "namespace": NAMESPACE, "labels": {"app": "bhanu-hallumat"}},
        "spec": {
            "backoffLimit": 3,
            "activeDeadlineSeconds": hours * 3600,
            "ttlSecondsAfterFinished": 7 * 24 * 3600,
            "template": {"metadata": {"labels": {"app": "bhanu-hallumat"}}, "spec": pod},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--ref", required=True, help="git commit or branch of HalluMat to install")
    parser.add_argument("--model")
    parser.add_argument("--shard", help="INDEX/COUNT")
    parser.add_argument("--tensor-parallel", type=int, default=2, help="GPUs for the writer model")
    parser.add_argument("--name-suffix", default="", help="appended to the Job name, e.g. -2 for a rerun")
    parser.add_argument("passthrough", nargs="*", help="extra arguments for `hallumat extend` (after --)")
    args = parser.parse_args(argv)
    json.dump(render(args), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
