"""FastAPI server exposing the TypeSafe-compatible endpoint `POST /v1/systemone`.

    reflex-serve --model Qwen/Qwen3-8B --port 8008

Then any client written for Jev can point at http://localhost:8008 instead.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from reflex.schema import SystemOneRequest, SystemOneResponse

log = logging.getLogger("reflex.server")


def create_app(engine, api_key: str | None = None) -> FastAPI:
    """The HTTP surface. `api_key` (or env REFLEX_API_KEY) makes /v1/systemone require
    `Authorization: Bearer <key>`, which you want on any endpoint reachable from the
    internet, e.g. a rented GPU an evaluator calls."""
    app = FastAPI(title="reflex", version="0.1.0")
    api_key = api_key or os.environ.get("REFLEX_API_KEY") or None

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if (
            api_key
            and request.url.path.startswith("/v1/")
            and request.headers.get("authorization", "") != f"Bearer {api_key}"
        ):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid or missing API key",
                        "type": "invalid_api_key",
                    }
                },
            )
        return await call_next(request)

    @app.get("/healthz")
    @app.get("/health")
    def healthz():
        return {
            "ok": True,
            "status": "healthy",
            "model": engine.model_name,
            "calibration": engine.cal.temperature,
            "strategy": engine.strategy,
            "device": str(engine.device),
        }

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": engine.model_name, "object": "model", "owned_by": "reflex"}],
        }

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(req: SystemOneRequest):
        t0 = time.perf_counter()
        try:
            resp = engine.answer(req)
        except ValueError as e:  # bad labels / too long
            raise HTTPException(status_code=422, detail=str(e))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(status_code=529, detail="overloaded: request too large for GPU")
        ms = (time.perf_counter() - t0) * 1000
        log.info(
            "%d questions, %d state tok (%s), %d q tok, %.0f ms",
            len(req.questions),
            resp.usage.state_tokens,
            "hit" if resp.usage.state_cache_hit else "miss",
            resp.usage.question_tokens,
            ms,
        )
        return JSONResponse(resp.model_dump(), headers={"x-reflex-latency-ms": f"{ms:.1f}"})

    return app


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument(
        "--adapter", default=None, help="LoRA adapter: a reflex-calibrate output dir or a hub id"
    )
    ap.add_argument(
        "--calibration",
        default=None,
        help="calibration.json (default: the one next to the adapter)",
    )
    ap.add_argument(
        "--api-key", default=None, help="require this bearer key on /v1/* (or env REFLEX_API_KEY)"
    )
    ap.add_argument(
        "--served-name", default=None, help="name reported in responses (default: the model id)"
    )
    ap.add_argument("--prompt-style", default="markdown", choices=["markdown", "compact"])
    ap.add_argument(
        "--prompt-texts",
        default=None,
        help="prompt.json from reflex-optimize (instruction wording)",
    )
    ap.add_argument(
        "--permutations",
        type=int,
        default=1,
        help="default option-order averaging for requests that do not set it "
        "(2 halves letter-position bias at 2x branch cost)",
    )
    ap.add_argument(
        "--stable",
        action="store_true",
        help="take adapter/calibration/prompt defaults from serving/stable.json "
        "(the configuration the `stable` git tag recommends); explicit flags still win",
    )
    ap.add_argument(
        "--think",
        type=int,
        default=0,
        help="System Two readout: reasoning tokens per branch before the logits are read "
        "(slow; for offline teachers and experiments, not serving)",
    )
    ap.add_argument(
        "--ensemble", default=None, help="prompt-ensemble variants json (reflex.ensemble)"
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--max-pack-tokens", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    import uvicorn

    from reflex.engine import Engine

    if args.stable:
        from reflex.serving import engine_kwargs, load_stable

        kw = engine_kwargs(
            load_stable(),
            adapter_path=args.adapter,
            calibration_path=args.calibration,
            prompt_texts=args.prompt_texts,
            prompt_style=args.prompt_style if args.prompt_style != "markdown" else None,
            default_permutations=args.permutations if args.permutations != 1 else None,
        )
        if args.model != ap.get_default("model"):
            kw["model_id"] = args.model
    else:
        kw = {
            "model_id": args.model,
            "calibration_path": args.calibration,
            "adapter_path": args.adapter,
            "default_permutations": args.permutations,
            "prompt_style": args.prompt_style,
            "prompt_texts": args.prompt_texts,
        }
    engine = Engine.load(
        dtype=getattr(torch, args.dtype),
        max_pack_tokens=args.max_pack_tokens,
        think_tokens=args.think,
        ensemble=args.ensemble,
        **kw,
    )
    if args.served_name:
        engine.model_name = args.served_name
    uvicorn.run(
        create_app(engine, api_key=args.api_key),
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
