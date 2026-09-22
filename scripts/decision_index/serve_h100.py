"""Supervise the pinned Reflex and SGLang servers in one CentML GPU container."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

MODEL = "Qwen/Qwen3.8-27B-FP8"
REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
SOURCE = "e21b3b23afdfeee7021a6604fa38f57e7ff5187f"


def commands(snapshot, *, permutations=2, mamba_slots=160, ssm_dtype=None):
    # Historical harnesses import this helper; preserve their original defaults.
    engine = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        snapshot,
        "--served-model-name",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "30000",
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        "0.85",
        "--context-length",
        "65536",
        "--max-running-requests",
        "32",
        "--max-total-tokens",
        "65536",
        "--max-mamba-cache-size",
        str(mamba_slots),
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--cuda-graph-max-bs-decode",
        "32",
        "--chunked-prefill-size",
        "8192",
        "--attention-backend",
        "fa3",
        "--enable-metrics",
    ]
    front = [
        sys.executable,
        "-m",
        "reflex.server",
        "--backend",
        "sglang",
        "--sglang-url",
        "http://127.0.0.1:30000",
        "--model",
        snapshot,
        "--permutations",
        str(permutations),
        "--served-name",
        "reflex-27b",
        "--host",
        "0.0.0.0",
        "--port",
        "8008",
    ]
    if ssm_dtype is not None:
        engine += ["--mamba-ssm-dtype", ssm_dtype]
    return engine, front


def deployment_commands(snapshot):
    return commands(snapshot, permutations=2, mamba_slots=320, ssm_dtype="bfloat16")


def wait_ready(url, child, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"Server exited during startup: {child.returncode}")
        try:
            if httpx.get(url, timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise TimeoutError(f"Server was not ready within {timeout}s")


def supervise(engine_command, front_command):
    children = []
    try:
        engine = subprocess.Popen(engine_command, start_new_session=True)
        children.append(engine)
        wait_ready("http://127.0.0.1:30000/health", engine, 1200)
        front = subprocess.Popen(front_command, start_new_session=True)
        children.append(front)
        wait_ready("http://127.0.0.1:8008/healthz", front, 120)
        print("REFLEX_SERVICE_READY", flush=True)
        while True:
            for child in children:
                if child.poll() is not None:
                    raise RuntimeError(f"Serving process exited: {child.returncode}")
            time.sleep(2)
    finally:
        # Both process groups include any SGLang workers. Do not leave the API
        # healthy while its inference process has exited.
        for child in reversed(children):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()


def main():
    from huggingface_hub import snapshot_download

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    os.environ["PYTHONPATH"] = str(Path(__file__).parent / "src")
    print(json.dumps({"source": SOURCE, "model": MODEL, "revision": REVISION}), flush=True)
    snapshot = snapshot_download(
        MODEL,
        revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )
    supervise(*deployment_commands(snapshot))


if __name__ == "__main__":
    main()
