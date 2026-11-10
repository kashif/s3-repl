"""Write swarmlab specs for the s3 replication (PLAN.md): information aggregation, s3_agent.

    python gen_specs.py --name eval_base --ns 4 --seeds 1-192
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

MODEL = "Qwen/Qwen3-0.6B"
BASE_URL = "http://127.0.0.1:8000"  # rewritten to the job's port by slurm/serve_and_run.sbatch


def arm(n: int, params: dict) -> dict:
    # module:Class names, not entry points: the trainer imports swarmlab from PYTHONPATH (no installed dist)
    return {
        "world": {"type": "swarmlab.world.aggregation:Aggregation"},
        "medium": {"topology": {"type": "swarmlab.medium.topology:Direct", "params": {"allow_all": True}}, "delivery": "pull"},
        "participants": [{"type": "swarmlab.participants.s3:S3Agent", "count": n, "role": "s3",
                          "params": {"model": MODEL, "base_url": BASE_URL, **params}}],
    }


def seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ns", type=int, nargs="+", default=[4])
    ap.add_argument("--seeds", default="1-192")
    ap.add_argument("--work-tokens", type=int, default=8192)
    a = ap.parse_args()
    spec = {
        "name": f"s3-{a.name}",
        "seeds": seeds(a.seeds),
        "budget": {"soft_usd": 0, "hard_usd": 0},
        "options": {"commit": "async", "rounds_per_agent": 1, "max_calls_per_turn": 100000,
                    "concurrency": 1024},
        "roles": {"s3": {"tools": ["post", "read_board", "submit", "wait"]}},
        "arms": {f"agg_n{n}": arm(n, {"work_tokens": a.work_tokens}) for n in a.ns},
    }
    out = Path(__file__).parent / "specs" / f"{a.name}.yaml"
    out.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"wrote {out} ({len(spec['arms'])} arms x {len(spec['seeds'])} seeds)")


if __name__ == "__main__":
    main()
