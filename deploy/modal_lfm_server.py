"""Serve LFM2.5-2.6B on Modal as a pay-as-you-go, OpenAI-compatible endpoint.

Modal runs the vLLM OpenAI server on a GPU that **scales to zero** when idle, so
you pay only while it is actually serving. The NextSearch harness then drives it
remotely over HTTP — set `NEXTSEARCH_BASE_URL` to this endpoint and run rollouts
from your laptop (CPU is fine) or a notebook, with no local GPU and nothing to
re-install each session.

Modeled on the official vLLM recipe for LFM2.5
(github.com/vllm-project/recipes, LiquidAI/lfm25-modal.py); check that recipe
and the Modal docs if a decorator signature has moved on.

Prerequisites (one time):

    pip install modal
    modal setup                                   # authenticate this machine
    modal secret create huggingface HF_TOKEN=hf_… # for model download / rate limits

Deploy a persistent, scale-to-zero endpoint:

    modal deploy deploy/modal_lfm_server.py
    # override the model or GPU:
    MODEL=LiquidAI/LFM2.5-8B-A1B GPU=H100 modal deploy deploy/modal_lfm_server.py

Modal prints a URL like
`https://<workspace>--nextsearch-lfm-vllm-serve.modal.run`. Point NextSearch at
it (note the trailing `/v1`):

    export NEXTSEARCH_BASE_URL="https://<workspace>--nextsearch-lfm-vllm-serve.modal.run/v1"
    nextsearch-compat --model lfm2.5-2.6b
    nextsearch-eval rollout --benches seal0:20 --models lfm2.5-2.6b

Optional auth: attach a `vllm-api-key` secret holding `VLLM_API_KEY`, and this
server passes it to vLLM as `--api-key`. Set the same `VLLM_API_KEY` locally —
the NextSearch vllm client already reads it.
"""

import os

import modal

MODEL_NAME = os.environ.get("MODEL", "LiquidAI/LFM2.5-2.6B")
GPU = os.environ.get("GPU", "L4")            # LFM2.5-2.6B is tiny; L4 is plenty
VLLM_PORT = 8000
MINUTES = 60

# Stage-1 serving defaults, matching the notebook/docs. A model window of 32k
# with the harness policy cap below it; the lfm2 tool parser is mandatory for
# tool calling.
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "32"))
GPU_MEM_UTIL = os.environ.get("GPU_MEMORY_UTILIZATION", "0.90")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm>=0.23.0", "huggingface_hub[hf_transfer]>=0.28")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

app = modal.App("nextsearch-lfm-vllm")

# Cache weights and the vLLM compile cache across cold starts so scale-from-zero
# is fast and cheap after the first boot.
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


def _secrets():
    """Attach optional secrets only when they exist, so a plain deploy works
    for the public LFM2.5-2.6B without any secret setup."""
    out = []
    for name in ("huggingface", "vllm-api-key"):
        try:
            out.append(modal.Secret.from_name(name))
        except Exception:  # noqa: BLE001 — secret not created; that is fine
            pass
    return out


@app.function(
    image=vllm_image,
    gpu=f"{GPU}:1",
    scaledown_window=10 * MINUTES,   # pay-as-you-go: idle -> zero after 10 min
    timeout=20 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache,
             "/root/.cache/vllm": vllm_cache},
    secrets=_secrets(),
)
@modal.concurrent(max_inputs=MAX_NUM_SEQS)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        # served-model-name matches the registry model_id, so the NextSearch
        # `lfm2.5-2.6b` entry works unchanged against this endpoint.
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--enable-auto-tool-choice", "--tool-call-parser", "lfm2",
        "--uvicorn-log-level=info",
    ]
    # Optional bearer-token auth when a vllm-api-key secret is attached.
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        cmd += ["--api-key", api_key]

    print("launching:", " ".join(cmd))
    subprocess.Popen(cmd)
