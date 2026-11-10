"""Decode one agent's token context (what the model saw and wrote) from `eval.py --save-tokens` output.

    python analysis/transcript.py runs/base.jsonl PROBLEM AGENT_NAME [--model Qwen/Qwen3-0.6B]
"""
import argparse
import json

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("jsonl")
ap.add_argument("problem", type=int)
ap.add_argument("agent")
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
a = ap.parse_args()
ep = next(r for r in map(json.loads, open(a.jsonl)) if r["problem"] == a.problem)
ag = next(x for x in ep["agents"] if x["name"] == a.agent)
toks, mask = ag["tokens"], ag["mask"]
tok = AutoTokenizer.from_pretrained(a.model)
i = 0
while i < len(toks):  # runs of model (mask 1) and environment (mask 0) tokens, decoded separately
    j = i
    while j < len(toks) and mask[j] == mask[i]:
        j += 1
    print(("MODEL" if mask[i] else "ENV") + f" [{j - i} tok]:\n{tok.decode(toks[i:j])}\n")
    i = j
