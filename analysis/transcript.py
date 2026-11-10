"""Decode one agent's token context (what the model saw and wrote) via the vLLM server.

    python analysis/transcript.py RUN_DIR AGENT PORT
"""
import json
import pickle
import sys
from pathlib import Path

import httpx

run, agent, port = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
snap = json.loads(sorted((run / "snapshots").glob("*.json"))[-1].read_text())
h = snap["plugins"][f"participant:{agent}"]
st = pickle.loads((run / "blobs" / h[:2] / h).read_bytes())
toks, mask = st["tokens"], st["mask"]
out, i = [], 0
while i < len(toks):  # decode runs of model (mask 1) and environment (mask 0) tokens separately
    j = i
    while j < len(toks) and mask[j] == mask[i]:
        j += 1
    text = httpx.post(f"http://127.0.0.1:{port}/detokenize", json={"model": "Qwen/Qwen3-0.6B",
                                                                 "tokens": toks[i:j]}).json()["prompt"]
    out.append(("MODEL" if mask[i] else "ENV") + f" [{j - i} tok]:\n{text}")
    i = j
print("\n\n".join(out))
