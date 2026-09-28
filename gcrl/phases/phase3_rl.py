"""Phase 3: offline policy optimisation.

This is where the pipeline is actually connected. Two wiring decisions define
the paper's contribution and both are explicit config switches, so the ablation
is two real runs rather than two labels on one run:

``rl.state_source``
    ``"gnn_embeddings"`` builds the agent's state from the Phase 1 embedding of
    each round's user. ``"raw_features"`` uses the tabular columns. The previous
    implementation described the former and implemented the latter -- KuaiRec
    states were ``['video_duration', 'timestamp']`` -- so no result could be
    attributed to graph structure.

``rl.cate_reward_weight`` (lambda)
    Shapes the reward as ``r = base_reward + lambda * CATE(x)``. At ``lambda=0``
    the agent is the same architecture with an unshaped reward, which is the
    controlled comparison for "does causal reward shaping help?". Phase 2's CATE
    output was previously never read by any downstream phase.

Every agent is trained and evaluated by the same code path. No agent's score is
written as a literal.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig, resolve_device
from ..logging_utils import get_logger
from ..models.rl import BasePolicy, D3RLPyPolicy, build_policy
from .phase1_gnn import embedding_path, load_embeddings

logger = get_logger(__name__)


@dataclass
class PolicyTrainingResult:
    agent: str
    dataset: str
    state_source: str
    state_dim: int
    n_actions: int
    n_train_rounds: int
    cate_reward_weight: float
    training_seconds: float
    seed: int
    policy_path: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def build_state(
    config: ExperimentConfig,
    raw_features: np.ndarray,
    user_index: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    """Assemble the agent's observation according to ``rl.state_source``.

    Raises:
        ValueError: if embeddings are requested without a user index, or if the
            index addresses rows outside the embedding matrix. Neither is
            recoverable by substituting raw features, because that would change
            what the reported result measures.
    """
    if config.rl.state_source == "raw_features":
        logger.info("State = raw tabular features, %d dimensions", raw_features.shape[1])
        return np.nan_to_num(raw_features.astype(np.float32))

    if user_index is None:
        raise ValueError(
            "state_source='gnn_embeddings' requires a per-round user index to look up "
            "each round's embedding"
        )

    path = embedding_path(config, config.rl.gnn_architecture, seed)
    payload = load_embeddings(path)
    try:
        state = payload.for_users(user_index)
    except IndexError as exc:
        raise ValueError(
            f"{exc}. The embedding table in {path.name} and the interaction log disagree; "
            f"rebuild the graph and rerun Phase 1 rather than clipping the index."
        ) from exc
    logger.info(
        "State = %s embeddings from %s, %d dimensions",
        config.rl.gnn_architecture, path.name, state.shape[1],
    )
    return state


def shape_rewards(
    base_rewards: np.ndarray,
    cate: np.ndarray | None,
    weight: float,
) -> np.ndarray:
    """Apply causal reward shaping ``r + lambda * CATE``.

    Raises:
        ValueError: if a non-zero weight is requested without CATE estimates.
            Falling back to the unshaped reward would make the ablation compare
            a configuration against itself.
    """
    if weight == 0.0:
        logger.info("Reward shaping disabled (lambda = 0); using the base reward")
        return base_rewards.astype(np.float64)

    if cate is None:
        raise ValueError(
            f"cate_reward_weight={weight} requires CATE estimates from Phase 2, but none "
            f"were supplied. Run Phase 2 first, or set the weight to 0."
        )
    if len(cate) != len(base_rewards):
        raise ValueError(f"CATE has {len(cate)} entries but there are {len(base_rewards)} rounds")

    shaped = base_rewards.astype(np.float64) + weight * np.asarray(cate, dtype=np.float64)
    logger.info(
        "Reward shaped with lambda=%.4g | base mean %.6f -> shaped mean %.6f "
        "(CATE mean %.6f, std %.6f)",
        weight, base_rewards.mean(), shaped.mean(), np.mean(cate), np.std(cate),
    )
    return shaped


def train_single_policy(
    agent_name: str,
    state: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_actions: int,
    config: ExperimentConfig,
    seed: int,
    device: str,
) -> tuple:
    """Train one agent and persist it. Returns ``(policy, result)``."""
    started = time.time()
    policy = build_policy(agent_name, n_actions, state.shape[1], config.rl, device, seed)

    logger.info(
        "[%s] training %s | %d rounds, state dim %d, %d actions",
        config.dataset.name, agent_name, len(actions), state.shape[1], n_actions,
    )
    policy.fit(state, actions, rewards)

    policy_dir = Path(config.paths.policies)
    policy_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_lambda{config.rl.cate_reward_weight:g}" if config.rl.cate_reward_weight else ""
    policy_path = (
        policy_dir
        / f"{config.dataset.name}_{agent_name}_{config.rl.state_source}{suffix}_seed{seed}.pt"
    )
    policy.save(policy_path)

    result = PolicyTrainingResult(
        agent=agent_name,
        dataset=config.dataset.name,
        state_source=config.rl.state_source,
        state_dim=int(state.shape[1]),
        n_actions=n_actions,
        n_train_rounds=len(actions),
        cate_reward_weight=config.rl.cate_reward_weight,
        training_seconds=time.time() - started,
        seed=seed,
        policy_path=str(policy_path),
    )
    logger.info("[%s] %s trained in %.1fs", config.dataset.name, agent_name, result.training_seconds)
    return policy, result


def run_phase3(
    state: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_actions: int,
    config: ExperimentConfig,
    seed: int | None = None,
) -> dict[str, BasePolicy]:
    """Train every configured agent on the same data with the same protocol."""
    seed = seed if seed is not None else config.seed
    device = resolve_device(config.device)
    logger.info(
        "Phase 3 on %s | agents: %s | state=%s (%d dim) | lambda=%g",
        device, config.rl.agents, config.rl.state_source, state.shape[1],
        config.rl.cate_reward_weight,
    )

    policies: dict[str, BasePolicy] = {}
    results: list[PolicyTrainingResult] = []

    for agent_name in config.rl.agents:
        policy, result = train_single_policy(
            agent_name, state, actions, rewards, n_actions, config, seed, device
        )
        policies[agent_name] = policy
        results.append(result)

    output = Path(config.paths.results) / f"phase3_rl_{config.experiment_name}_seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump([r.as_row() for r in results], handle, indent=2)
    logger.info("Phase 3 training metadata -> %s", output)

    return policies


def load_policy(
    agent_name: str,
    path: Path,
    n_actions: int,
    state_dim: int,
    config: ExperimentConfig,
    seed: int,
    device: str,
    sample_state: np.ndarray | None = None,
    sample_actions: np.ndarray | None = None,
) -> BasePolicy:
    """Reconstruct a trained policy from disk."""
    policy = build_policy(agent_name, n_actions, state_dim, config.rl, device, seed)
    if isinstance(policy, D3RLPyPolicy):
        if sample_state is None or sample_actions is None:
            raise ValueError(
                f"{agent_name} needs a sample batch to rebuild its network graph before loading"
            )
        policy.build_for_loading(sample_state, sample_actions)
    policy.load(path)
    return policy
