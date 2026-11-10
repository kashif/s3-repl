"""s3 training (arXiv 2611.08615, §3.2-3.4, App. A, C, I) with TRL's AsyncGRPOTrainer and the bespoke harness in `s3/`.

One TRL rollout is one whole team episode: `generate_one` awaits `s3.run_team` (N token-level agents against the
vLLM server, an in-memory board) on the rollout loop's own event loop, and returns every agent's sampled stretches
as token turns. Each agent's turns chain into one training row, so a team gives N rows, all with the team's advantage.

The paper's objective, information aggregation:
- Team reward (Eq. 1): R = S * (1 - 0.25 * min(1, sum_i |tau_i| / (N L))), S = any agent's final
  list correct, |tau_i| = tokens agent i generated, L = 8,192 work + 1,024 final.
- Team-relative advantage (Eq. 2): `num_generations` = G = 8 team attempts per problem;
  A_g = (R_g - mean R) / (std R + 1e-8), TRL's default `compute_advantages`; a group whose teams all score the
  same is kept as advantage-0 rows (App. C: it still counts toward collection).
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
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from datasets import Dataset

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer, AsyncRolloutLoop, Rollout
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample, TurnRecord

from s3 import Settings, VLLMSampler, run_team

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("s3")


@dataclass
class S3Config(AsyncGRPOConfig):
    s3_agents: int = field(default=4)
    s3_work_tokens: int = field(default=8192)
    s3_final_tokens: int = field(default=1024)
    s3_model: str = field(default="Qwen/Qwen3-0.6B", metadata={"help": "served model name on the vLLM server"})


def agent_turns(state) -> list[TurnRecord]:
    """One TurnRecord per contiguous stretch of model-sampled tokens; the context before it is the prompt."""
    toks, mask, lps = state.tokens, state.mask, state.logprobs
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
    """Rollouts are `s3.run_team` episodes (see the module docstring)."""

    async def setup(self) -> None:
        c = self.config
        self._sampler = VLLMSampler(c.vllm_server_base_url, c.s3_model)
        self._settings = Settings(work_tokens=c.s3_work_tokens, final_tokens=c.s3_final_tokens)
        self._rng = random.Random(os.getpid())

    async def _init_concurrency_cap(self) -> None:
        """Start at `max_concurrent_rollouts` teams. TRL's default (KV cache tokens / max_model_len) assumes one
        full-length sequence per rollout, about 14 here; a team is N shorter agent contexts, many of them idle at a
        time. The control loop still cuts the cap when KV cache usage passes 0.8."""
        self._concurrency_cap = max(1, self.config.max_concurrent_rollouts)

    async def close(self) -> None:
        await self._sampler.aclose()

    async def generate_one(self, prompt, tools, group_id: int = 0) -> Rollout:
        problem = int(prompt[-1]["content"].split("problem:", 1)[1])
        n = self.config.s3_agents
        try:
            ep = await run_team(self._sampler, problem, self._rng.randrange(1 << 31), n, self._settings)
        except Exception as e:  # an infrastructure failure (TRL re-raises and stops): unscored, so no rows
            logger.warning("team episode for problem %d failed: %s: %s", problem, type(e).__name__, e)
            self._push_metrics({"s3/episode_errors": 1.0})
            return Rollout(completion=[{"role": "assistant", "content": f"error {type(e).__name__}"}],
                           completion_ids=[], turns=[], reward=None)
        turns, completion_ids = [], []
        for st in ep.agents:
            t = agent_turns(st)
            turns += t
            completion_ids += [x for tr in t for x in tr.output_ids]
        msgs = sum(e["kind"] == "message" for st in ep.agents for e in st.events)
        self._push_metrics({"s3/success": float(ep.success), "s3/team_tokens": float(ep.tokens),
                            "s3/messages": float(msgs)})
        summary = [f"problem {problem} target {ep.problem.target} success {ep.success} reward {ep.reward:.3f}"]
        summary += [f"{st.name}: {st.work_generated}+{st.final_generated} tok, "
                    f"{sum(e['kind'] == 'message' for e in st.events)} msgs, answer {st.answer}" for st in ep.agents]
        return Rollout(completion=[{"role": "assistant", "content": "\n".join(summary)}],
                       completion_ids=completion_ids, turns=turns, reward=ep.reward)

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
    ap.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--output-dir", default=str(HERE / "checkpoints" / "s3"))
    ap.add_argument("--concurrent-teams", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-7)
    a = ap.parse_args()
    n = 4
    rows = [{"prompt": [{"role": "user", "content": f"problem:{i}"}]} for i in range(a.problems)]
    config = S3Config(
        output_dir=a.output_dir,
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
