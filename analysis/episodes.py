"""Per-episode summary of s3 runs: success, per-agent tokens, messages, reads, answers, and the
paper's Table 11 failure causes (first matching).

    uv run --project /fsx/kashif/swarmlab python analysis/episodes.py runs/ [--experiment s3-eval_base] [--show RUN]
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

from swarmlab.world.aggregation import number_words

LAMBDA, WORK, FINAL = 0.25, 8192, 1024


def blob(run: Path, h: str) -> bytes:
    return (run / "blobs" / h[:2] / h).read_bytes()


def load(run: Path) -> tuple[dict, dict]:
    meta = json.loads((run / "run.json").read_text())
    snaps = sorted((run / "snapshots").glob("*.json"))
    snap = json.loads(snaps[-1].read_text())
    agents = {}
    for k, h in snap["plugins"].items():
        if k.startswith("participant:"):
            agents[k.split(":", 1)[1]] = pickle.loads(blob(run, h))
    return meta, agents


def own_number_sent(st: dict, number: int) -> bool:
    words, digits = number_words(number), str(number)
    for e in st.get("events", []):
        if e["kind"] == "message" and e.get("ok"):
            t = e["text"].lower()
            if words in t or re.search(rf"(?<!\d){digits}(?!\d)", t):
                return True
    return False


def summarize(run: Path) -> dict | None:
    meta, agents = load(run)
    score = meta.get("score") or {}
    if not agents or "numbers" not in score:
        return None
    numbers, target = score["numbers"], sorted(score["numbers"].values())
    per = {}
    for a, st in sorted(agents.items()):
        ev = st.get("events", [])
        msgs = [e for e in ev if e["kind"] == "message"]
        per[a] = {"work": st.get("work_generated", 0), "final": st.get("final_generated", 0),
                  "messages": len(msgs), "broadcasts": sum(e["to"].lower() == "all" for e in msgs),
                  "reads": sum(e["kind"] == "read" for e in ev),
                  "sent_own": own_number_sent(st, numbers[a]), "answer": st.get("answer"),
                  "work_end": next((e["reason"] for e in ev if e["kind"] == "work_end"), None),
                  "read_before_send": next((e["kind"] == "read" for e in ev if e["kind"] in ("read", "message")),
                                           False)}
    success = bool(score.get("success"))
    tokens = sum(p["work"] + p["final"] for p in per.values())
    n = len(per)
    reward = (1 - LAMBDA * min(1.0, tokens / (n * (WORK + FINAL)))) if success else 0.0
    cause = None
    if not success:
        if any(p["messages"] == 0 for p in per.values()):
            cause = "a number withheld: an agent never messaged"
        elif not all(p["sent_own"] for p in per.values()):
            cause = "a number withheld: asked only, or misstated"
        elif not any(p["reads"] for p in per.values()):
            cause = "sent but never read"
        else:
            cause = "had or could have had all numbers, answer wrong"
    return {"run": meta.get("run") or run.name, "success": success, "reward": round(reward, 3),
            "tokens": tokens, "n": n, "target": target, "cause": cause, "agents": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--experiment")
    ap.add_argument("--show")
    a = ap.parse_args()
    rows = []
    for rj in sorted(a.runs.glob("*/run.json")):
        if a.experiment and not rj.parent.name.startswith(a.experiment + "__"):
            continue
        s = summarize(rj.parent)
        if s:
            rows.append(s)
    for s in rows:
        if a.show and a.show not in s["run"]:
            continue
        print(f"{s['run']}: success={s['success']} reward={s['reward']} tokens={s['tokens']} "
              f"target={s['target']} cause={s['cause']}")
        for ag, p in s["agents"].items():
            print(f"   {ag}: work={p['work']} final={p['final']} msgs={p['messages']} (all={p['broadcasts']}) "
                  f"reads={p['reads']} sent_own={p['sent_own']} read_first={p['read_before_send']} "
                  f"end={p['work_end']} answer={p['answer']}")
    if rows:
        k = len(rows)
        print(f"\n{k} episodes: success {sum(r['success'] for r in rows) / k:.1%}, "
              f"mean tokens {sum(r['tokens'] for r in rows) / k:.0f}")
        causes: dict[str, int] = {}
        for r in rows:
            if r["cause"]:
                causes[r["cause"]] = causes.get(r["cause"], 0) + 1
        fails = sum(causes.values())
        for c, v in sorted(causes.items(), key=lambda x: -x[1]):
            print(f"  {c}: {v / fails:.0%} of failures")


if __name__ == "__main__":
    main()
