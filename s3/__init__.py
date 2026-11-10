"""Bespoke s3 (arXiv 2611.08615) harness for information aggregation: task, prompts, token-level team episodes."""
from .task import Problem, parse_answer, team_reward
from .team import Board, Episode, Settings, VLLMSampler, run_team

__all__ = ["Board", "Episode", "Problem", "Settings", "VLLMSampler", "parse_answer", "run_team", "team_reward"]
