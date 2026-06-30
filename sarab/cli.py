"""SARAB command-line interface.

    sarab run   <model> --prompt "..."     one-shot streaming generation + stats
    sarab serve <model> [--port 8000]      OpenAI-compatible HTTP server
    sarab bench <model> [...]              throughput + peak-RAM benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

from .config import RuntimeConfig


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="HF model id or local checkpoint directory")
    p.add_argument("--quant", choices=["none", "int8", "int4"], default="int8")
    p.add_argument("--ram-budget", type=float, default=4.0, help="soft RAM ceiling (GB)")
    p.add_argument("--resident-layers", type=int, default=2)
    p.add_argument("--prefetch-depth", type=int, default=1)
    p.add_argument("--no-prefetch", action="store_true")
    p.add_argument("--max-context", type=int, default=4096)


def _config_from_args(a: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        ram_budget_gb=a.ram_budget,
        resident_layers=a.resident_layers,
        prefetch_depth=a.prefetch_depth,
        prefetch=not a.no_prefetch,
        quant=a.quant,
        max_context=a.max_context,
    )


def _cmd_run(a: argparse.Namespace) -> int:
    from .hub import load

    cfg = _config_from_args(a)
    print(f"[sarab] loading {a.model}  (quant={cfg.quant}, "
          f"resident={cfg.resident_layers}, prefetch={cfg.prefetch_depth})",
          file=sys.stderr)
    rt = load(a.model, config=cfg)
    try:
        full = rt.store.total_bytes() / 1024**3
        n_layers = rt.model_config.n_layers
        resident = rt.runtime.resident_layers
        mode = "all resident" if resident >= n_layers else f"streaming {resident}/{n_layers}"
        print(f"[sarab] model on disk: {full:.2f} GB | {mode} | "
              f"quant={cfg.quant}", file=sys.stderr)
        rt.runtime.prewarm(progress=True)

        t0 = time.perf_counter()
        n = 0
        for piece in rt.generate(
            a.prompt,
            max_new_tokens=a.max_new_tokens,
            temperature=a.temperature,
            top_p=a.top_p,
            stream=True,
        ):
            sys.stdout.write(piece)
            sys.stdout.flush()
            n += 1
        dt = time.perf_counter() - t0
        print(f"\n[sarab] {n} tokens in {dt:.1f}s = {n/max(dt,1e-9):.2f} tok/s "
              f"| streamer stats: {rt.runtime.streamer.stats}", file=sys.stderr)
    finally:
        rt.close()
    return 0


def _cmd_serve(a: argparse.Namespace) -> int:
    from .server.api import serve

    cfg = _config_from_args(a)
    serve(a.model, cfg, host=a.host, port=a.port)
    return 0


def _cmd_bench(a: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    cfg = _config_from_args(a)
    run_benchmark(a.model, cfg, prompt=a.prompt, max_new_tokens=a.max_new_tokens)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sarab", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="one-shot generation")
    _add_common(run)
    run.add_argument("--prompt", required=True)
    run.add_argument("--max-new-tokens", type=int, default=64)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.set_defaults(func=_cmd_run)

    serve = sub.add_parser("serve", help="OpenAI-compatible server")
    _add_common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    bench = sub.add_parser("bench", help="throughput + peak-RAM benchmark")
    _add_common(bench)
    bench.add_argument("--prompt", default="The capital of Egypt is")
    bench.add_argument("--max-new-tokens", type=int, default=64)
    bench.set_defaults(func=_cmd_bench)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
