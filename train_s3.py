"""s3 training (arXiv 2611.08615, §3.2-3.4, App. A, C, I) with TRL's AsyncGRPOTrainer and swarmlab rollouts.

One TRL rollout is one whole team episode: `generate_one` runs a swarmlab `aggregation` episode
(N `s3_agent`s against the vLLM server, `commit: async`) and returns every agent's sampled
stretches as token turns. Each agent's turns chain into one training row, so a team gives N rows,
all stamped with the team's advantage.

The paper's objective, information aggregation:
- Team reward (Eq. 1): R = S * (1 - 0.25 * min(1, sum_i |tau_i| / (N L))), S = any agent's final
  list correct, |tau_i| = tokens agent i generated, L = 8,192 work + 1,024 final.
- Team-relative advantage (Eq. 2): `num_generations` = G = 8 team attempts per problem;
  A_g = (R_g - mean R) / (std R + 1e-8), TRL's default `compute_advantages` (a group whose teams all
  score the same gets advantage 0 and is dropped; it still counts toward collection, as in App. C).
- Loss (Eq. 3): -1/|B| sum_g 1/|T_g| sum_{t in T_g} m_t rho_t A_g, rho_t = pi_theta / mu (mu the
  vLLM sampler's logprob), m_t = 1[0.2 <= rho_t <= 5.0]. Here: the importance mask is
  `vllm_importance_sampling_clip_min/max` = 0.2 / 5.0; PPO clipping is switched off (`epsilon` = 1,
  `epsilon_high` = 1e9, so the min of the clipped and unclipped terms is rho * A); micro-batches
  hold exactly one team (`per_device_train_batch_size` = N rows, fixed-count batching), so TRL's
  per-micro-batch token mean is the team mean 1/|T_g| sum_t; `gradient_accumulation_steps` = teams
  per update averages over teams (1/|B| sum_g).
- Not implemented: the KL penalty listed for aggregation in Table 4 (its coefficient and reference
  are not published).

Run (2 GPUs; see slurm/train.sbatch):
    CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --max-model-len 40960 \
        --logprobs-mode processed_logprobs --weight-transfer-config '{"backend":"nccl"}'
    CUDA_VISIBLE_DEVICES=1 accelerate launch train_s3.py --teams-per-update 128 --updates 50
"""
from __future__ import annotations

import argparse
import math
import asyncio
import json
import os
import pickle
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from datasets import Dataset

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer, AsyncRolloutLoop, Rollout
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample, TurnRecord

LAMBDA = 0.25
HERE = Path(__file__).resolve().parent


@dataclass
class S3Config(AsyncGRPOConfig):
    s3_spec: str = field(default=str(HERE / "specs" / "train.yaml"), metadata={"help": "swarmlab spec (one arm)"})
    s3_arm: str = field(default="agg_n4")
    s3_agents: int = field(default=4)
    s3_allowance: int = field(default=8192 + 1024, metadata={"help": "L in Eq. 1: work + final tokens per agent"})
    s3_individual_reward: bool = field(default=False, metadata={"help": "control: reward each agent for its own answer"})


def team_reward(success: bool, tokens: int, n: int, allowance: int) -> float:
    """Eq. 1."""
    return (1.0 - LAMBDA * min(1.0, tokens / (n * allowance))) if success else 0.0


def agent_turns(state: dict) -> list[TurnRecord]:
    """One TurnRecord per contiguous stretch of model-sampled tokens; the context before it is the prompt."""
    toks, mask, lps = state["tokens"], state["mask"], state["logprobs"]
    turns, i = [], 0
    while i < len(toks):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(toks) and mask[j]:
            j += 1
        turns.append(TurnRecord(prompt_ids=list(toks[:i]), output_ids=list(toks[i:j]),
                                output_log_probs=[float(x) for x in lps[i:j]]))
        i = j
    return turns


class S3RolloutLoop(AsyncRolloutLoop):
    """Rollouts are swarmlab team episodes (see the module docstring)."""

    async def setup(self) -> None:
        from swarmlab import Experiment

        self._proto = Experiment.from_yaml(self.config.s3_spec, self.config.s3_arm)
        self._rng = random.Random(os.getpid())
        self._tmp = Path(os.environ.get("S3_TMP") or tempfile.gettempdir()) / f"s3-rollouts-{os.getpid()}"
        self._tmp.mkdir(parents=True, exist_ok=True)
        # one thread per team in flight: `asyncio.to_thread`'s default pool (32 workers) capped the teams at 32
        self._episodes = ThreadPoolExecutor(max_workers=max(1, self.config.max_concurrent_rollouts),
                                            thread_name_prefix="s3-episode")

    async def _init_concurrency_cap(self) -> None:
        """Start at `max_concurrent_rollouts` teams. TRL's default (KV cache tokens / max_model_len) assumes one
        full-length sequence per rollout, about 14 here; a team is N shorter agent contexts, many of them idle at a
        time. The control loop still cuts the cap when KV cache usage passes 0.8."""
        self._concurrency_cap = max(1, self.config.max_concurrent_rollouts)

    async def close(self) -> None:
        self._episodes.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _episode(self, problem: int) -> dict:
        from swarmlab.world.aggregation import Aggregation

        n = self.config.s3_agents
        exp = self._proto.model_copy(update={"world": Aggregation(problem=problem)})
        run_dir = Path(tempfile.mkdtemp(dir=self._tmp))
        try:
            run = exp.run(self._rng.randrange(1 << 31), max_rounds=n, commit="async", max_calls_per_turn=100000,
                          concurrency=1024, run_dir=run_dir / "run")
            d = Path(run.dir)
            meta = json.loads((d / "run.json").read_text())
            snap = json.loads(sorted((d / "snapshots").glob("*.json"))[-1].read_text())
            agents = {}
            for k, h in snap["plugins"].items():
                if k.startswith("participant:"):
                    agents[k.split(":", 1)[1]] = pickle.loads((d / "blobs" / h[:2] / h).read_bytes())
            return {"status": meta.get("status"), "score": meta.get("score") or {}, "agents": agents}
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    async def generate_one(self, prompt, tools, group_id: int = 0) -> Rollout:
        text = prompt[-1]["content"]
        problem = int(text.split("problem:", 1)[1])
        ep = await asyncio.get_running_loop().run_in_executor(self._episodes, self._episode, problem)
        agents, score = ep["agents"], ep["score"]
        n, ok = self.config.s3_agents, len(ep["agents"]) == self.config.s3_agents and "success" in ep["score"]
        turns, completion_ids, summary = [], [], []
        tokens = 0
        for a in sorted(agents):
            st = agents[a]
            t = agent_turns(st)
            turns += t
            completion_ids += [x for tr in t for x in tr.output_ids]
            gen = int(st.get("work_generated", 0)) + int(st.get("final_generated", 0))
            tokens += gen
            msgs = [e for e in st.get("events", []) if e["kind"] == "message"]
            summary.append(f"{a}: {gen} tok, {len(msgs)} msgs, answer {st.get('answer')}")
        success = bool(score.get("success"))
        if not ok:  # an infrastructure failure: unscored (NaN), so advantage 0 and no pull either way
            reward = None
        elif self.config.s3_individual_reward:
            reward = None  # TODO individual-reward control: needs per-agent rows rewarded separately
        else:
            reward = team_reward(success, tokens, n, self.config.s3_allowance)
        self._push_metrics({"s3/success": float(success), "s3/team_tokens": float(tokens),
                            "s3/messages": float(sum(len([e for e in st.get("events", []) if e["kind"] == "message"])
                                                     for st in agents.values()))})
        summary.insert(0, f"problem {problem} target {sorted(score.get('numbers', {}).values())} "
                          f"success {success} reward {reward}")
        return Rollout(completion=[{"role": "assistant", "content": "\n".join(summary)}],
                       completion_ids=completion_ids, turns=turns, reward=reward)


    async def _score_group(self, group):
        """Keep a group whose teams all got the same reward, as advantage-0 rows (Appendix C: such groups "have zero
        advantage; they still count toward the collection budget"). TRL drops them and generates more groups until an
        update has `gradient_accumulation_steps` teams with signal; kept, they add no gradient but count in the 1/|B|
        team average, so an update is exactly `teams_per_update` teams collected, as in the paper."""
        samples = await super()._score_group(group)
        if samples or group.rewards is None or not len(group.completions_sequences):
            return samples
        out = []
        for prompt, completion, sequences in zip(group.prompts, group.completions, group.completions_sequences,
                                                 strict=True):
            for seq in sequences:
                n = len(seq.input_ids)
                out.append(RolloutSample(
                    prompt=prompt, completion=completion,
                    input_ids=np.asarray(seq.input_ids, dtype=np.int64),
                    completion_mask=np.asarray(seq.completion_mask, dtype=np.int64),
                    old_log_probs=np.asarray(seq.old_log_probs, dtype=np.float32),
                    sampling_mask=seq.sampling_mask,
                    sampler_ids=np.full((n, 0), -1, dtype=np.int64),
                    sampler_log_probs=np.full((n, 0), -math.inf, dtype=np.float32),
                    advantage=0.0, model_version=group.model_version, group_id=group.group_id,
                    metrics={"reward_kept": float(np.nanmean(group.rewards)) if not np.isnan(group.rewards).all()
                             else 0.0}))
        return out


class S3Trainer(AsyncGRPOTrainer):
    loop_cls = S3RolloutLoop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams-per-update", type=int, default=128)
    ap.add_argument("--updates", type=int, default=50)
    ap.add_argument("--problems", type=int, default=1024, help="training problem ids 0..problems-1")
    ap.add_argument("--spec", default=str(HERE / "specs" / "train.yaml"))
    ap.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--output-dir", default=str(HERE / "checkpoints" / "s3"))
    ap.add_argument("--concurrent-teams", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-7)
    a = ap.parse_args()
    n = 4
    rows = [{"prompt": [{"role": "user", "content": f"problem:{i}"}]} for i in range(a.problems)]
    config = S3Config(
        output_dir=a.output_dir,
        s3_spec=a.spec,
        s3_agents=n,
        num_generations=8,                       # G team attempts per problem
        per_device_train_batch_size=n,           # one team per micro-batch (fixed-count batching)
        token_budget=0,
        gradient_accumulation_steps=a.teams_per_update,
        max_steps=a.updates,
        learning_rate=a.lr, lr_scheduler_type="constant", warmup_steps=0,
        weight_decay=0.0, max_grad_norm=1.0, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        epsilon=1.0, epsilon_high=1e9,           # no PPO clipping: the term is m * rho * A
        vllm_importance_sampling_clip_min=0.2, vllm_importance_sampling_clip_max=5.0,
        score_centering=False,
        max_staleness=4,
        max_concurrent_rollouts=a.concurrent_teams,
        vllm_server_base_url=a.vllm_url,
        bf16=True, gradient_checkpointing=True,
        logging_steps=1, save_strategy="steps", save_steps=10, save_total_limit=1, report_to="none",
        log_completions=True, num_completions_to_print=1,
        rollout_worker_timeout=1800.0,
    )
    trainer = S3Trainer(model="Qwen/Qwen3-0.6B", args=config, train_dataset=Dataset.from_list(rows),
                        reward_funcs=[])
    trainer.train()
    trainer.end()


if __name__ == "__main__":
    main()
