"""Reference policies and concentration diagnostics for a fully observed block.

The OPE tables cannot answer two questions a reviewer will ask first.

**"Is 0.709 good?"**  Only against a ceiling and a floor. Both are lookups on
the fully observed block, not estimates:

* ORACLE -- the best available action for each evaluation user. The highest
  value any policy could attain on this data.
* BEST CONSTANT -- one action for everyone, chosen to maximise the value. This
  is the strongest recommender that does not personalise at all, and on a dense
  matrix it is a much harder baseline than uniform-random.

**"Did the policy actually personalise?"**  A tabular state with one dimension
cannot distinguish users, so an agent handed one has no way to vary its action
and collapses to a constant. That is invisible in a value column -- a collapsed
policy that lands on a popular item scores well. This counts the distinct
actions each trained policy selects across the evaluation contexts, so
collapse is reported as a measurement rather than inferred from a coincidence.

Both parts run on the CONTEXT REPRESENTATIVES, not the rounds. A fully observed
block is an enumeration: every round of a user carries that user's state, so a
policy makes one decision per user, not one per row. Querying 1,411 states
instead of 4,676,570 is exact, not an approximation, and it is what keeps this
script small enough to run beside a training job.

Usage:
    python scripts/reference_policies.py --config configs/_generated_kuairec_main.yaml --seed 42
    python scripts/reference_policies.py --config configs/_generated_kuairec_ablation_raw.yaml --seed 42

Writes ``results/reference_policies_<experiment>_seed<seed>.json`` and prints a
table. Reads nothing it does not already need for Phase 4; writes no artefact
any other phase consumes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gcrl.cli import _build_exact_ground_truth, _load_dataset  # noqa: E402
from gcrl.config import load_config, resolve_device  # noqa: E402
from gcrl.evaluation.ope import DeterministicPolicy, UniformPolicy  # noqa: E402
from gcrl.seeding import seed_everything  # noqa: E402

logger = logging.getLogger("reference_policies")


def context_representatives(round_context_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One round index per distinct context, and how many rounds each carries.

    The block is an enumeration, so a context's rounds are replicas of one
    decision. The weights are what make a mean over representatives equal the
    mean over rounds that the exact estimator reports.
    """
    rows = np.asarray(round_context_id)
    _, first_index, counts = np.unique(rows, return_index=True, return_counts=True)
    order = np.argsort(first_index)
    return first_index[order], counts[order]


def weighted_masked_mean(values: np.ma.MaskedArray, weights: np.ndarray) -> tuple[float, int, int]:
    """Mean over the OBSERVED entries only, weighted by round count.

    Returns ``(value, rounds_scored, rounds_masked)``. Unobserved cells are
    excluded from the average rather than imputed as zero -- the same
    convention ``ExactRewardLookup`` uses, so these numbers sit in the same
    column as the agents' exact values without a footnote of their own.
    """
    mask = np.ma.getmaskarray(values)
    keep = ~mask
    scored = int(weights[keep].sum())
    masked = int(weights[mask].sum())
    if scored == 0:
        return float("nan"), 0, masked
    filled = np.asarray(values.filled(0.0), dtype=np.float64)
    return float((filled[keep] * weights[keep]).sum() / scored), scored, masked


def reference_policies(block: np.ma.MaskedArray, weights: np.ndarray) -> dict:
    """ORACLE and BEST CONSTANT, by lookup on the ``(n_contexts, n_actions)`` block."""
    oracle_per_context = block.max(axis=1)
    oracle, oracle_scored, oracle_masked = weighted_masked_mean(oracle_per_context, weights)

    # One column per action: what that single action would earn if handed to
    # every user. Columns are averaged over their observed cells only.
    w = weights.astype(np.float64)[:, None]
    observed = (~np.ma.getmaskarray(block)).astype(np.float64)
    numerator = (np.asarray(block.filled(0.0), dtype=np.float64) * observed * w).sum(axis=0)
    denominator = (observed * w).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_action = np.where(denominator > 0, numerator / denominator, np.nan)

    ranked = np.argsort(np.nan_to_num(per_action, nan=-np.inf))[::-1]
    best = int(ranked[0])
    return per_action, denominator, {
        "oracle": {
            "value": oracle,
            "rounds_scored": oracle_scored,
            "rounds_masked": oracle_masked,
            "note": "best available action per user; the ceiling on this block",
        },
        "best_constant": {
            "value": float(per_action[best]),
            "action": best,
            "rounds_scored": int(denominator[best]),
            "note": "one action for every user; the strongest non-personalised policy",
        },
        "best_constant_top5": [
            {"action": int(a), "value": float(per_action[a]), "rounds_observed": int(denominator[a])}
            for a in ranked[:5]
        ],
    }


def constant_rank(per_action: np.ndarray, action: int) -> int:
    """Where one action ranks among all constant policies, 1 = best."""
    values = np.nan_to_num(per_action, nan=-np.inf)
    return int((values > values[action]).sum()) + 1


def train_popularity_constants(
    train_actions: np.ndarray,
    train_rewards: np.ndarray,
    per_action: np.ndarray,
    denominator: np.ndarray,
    n_actions: int,
    min_count: int = 100,
) -> dict:
    """Constant policies a practitioner could actually deploy.

    ``best_constant`` is chosen by reading the evaluation block's own rewards,
    so it is a hindsight ceiling on constant policies in the same way the oracle
    is a ceiling on all policies -- nobody could pick it in advance. These two
    are chosen from the TRAINING LOG alone and are therefore the honest
    non-personalised baselines an agent has to beat:

    * ``most_interactions`` -- the item the log contains most often. The
      classical popularity baseline.
    * ``highest_logged_reward_rate`` -- the item with the best observed reward
      rate among those seen at least ``min_count`` times. The threshold matters:
      without it an item watched twice and liked twice wins with a rate of 1.0.

    On KuaiRec both are well defined despite the cold-item regime: the log holds
    no (evaluation user, evaluation item) pair, but it does hold 10.2M rows over
    those items contributed by other users.
    """
    counts = np.bincount(train_actions, minlength=n_actions).astype(np.float64)
    positives = np.bincount(
        train_actions, weights=np.asarray(train_rewards, dtype=np.float64), minlength=n_actions
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(counts > 0, positives / counts, np.nan)

    eligible = counts >= min_count
    if not eligible.any():
        eligible = counts > 0
    picks = {
        "most_interactions": int(np.argmax(counts)),
        "highest_logged_reward_rate": int(
            np.argmax(np.where(eligible, np.nan_to_num(rate, nan=-np.inf), -np.inf))
        ),
    }
    out = {"min_count_for_rate": min_count}
    for name, action in picks.items():
        out[name] = {
            "action": action,
            "exact_value": float(per_action[action]),
            "rank_among_constants": constant_rank(per_action, action),
            "log_interactions": int(counts[action]),
            "log_reward_rate": float(rate[action]) if counts[action] else float("nan"),
            "rounds_observed": int(denominator[action]),
        }
    return out


def per_item_feature(
    train_actions: np.ndarray, train_raw: np.ndarray, n_actions: int
) -> np.ndarray | None:
    """Each item's tabular feature, averaged over the log rows that used it.

    Returns ``None`` unless the state is a single column, since that is the only
    case where "the feature" names one thing.
    """
    raw = np.asarray(train_raw)
    if raw.ndim != 2 or raw.shape[1] != 1:
        return None
    feature = raw[:, 0].astype(np.float64)
    counts = np.bincount(train_actions, minlength=n_actions).astype(np.float64)
    sums = np.bincount(train_actions, weights=feature, minlength=n_actions)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def duration_baseline(per_item: np.ndarray, per_action: np.ndarray) -> np.ndarray:
    """What an item is worth GIVEN ONLY ITS DURATION.

    The relation is monotone but not linear -- reward here is
    ``1[play_duration / video_duration >= 2]``, so a shorter clip clears the bar
    more easily in a way that flattens out. An isotonic fit respects the
    monotonicity without assuming a shape, and it is what makes the residual
    below interpretable: whatever is left is NOT explained by length.
    """
    from sklearn.isotonic import IsotonicRegression

    usable = np.isfinite(per_item) & np.isfinite(per_action)
    model = IsotonicRegression(increasing=False, out_of_bounds="clip")
    model.fit(per_item[usable], per_action[usable])
    filled = np.where(np.isfinite(per_item), per_item, np.nanmedian(per_item[usable]))
    return np.asarray(model.predict(filled), dtype=np.float64)


def duration_adjusted_value(
    block: np.ma.MaskedArray,
    weights: np.ndarray,
    baseline: np.ndarray,
    actions: np.ndarray,
) -> dict:
    """A policy's value, and the part of it duration does not explain.

    ``predicted`` is what this policy would be worth if every item it picks were
    worth exactly what its length predicts. ``residual`` is the rest: positive
    means the policy chooses items that beat their own duration, which is the
    only evidence in this benchmark that a method has learned preference rather
    than length. A policy can score highly and still have a residual of zero --
    that is the whole point of computing it.
    """
    index = np.arange(len(actions))
    chosen = block[index, actions]
    realised, scored, masked = weighted_masked_mean(chosen, weights)
    predicted_col = np.ma.array(baseline[actions], mask=np.ma.getmaskarray(chosen))
    predicted, _, _ = weighted_masked_mean(predicted_col, weights)
    residual, _, _ = weighted_masked_mean(chosen - predicted_col, weights)
    return {
        "exact_value": realised,
        "duration_predicted": predicted,
        "duration_adjusted_residual": residual,
        "rounds_scored": scored,
        "rounds_masked": masked,
    }


def personalisation_test(
    block: np.ma.MaskedArray,
    weights: np.ndarray,
    actions: np.ndarray,
    n_permutations: int = 200,
    seed: int = 42,
) -> dict:
    """Does it matter WHICH user gets which item, or only which items are picked?

    The direct test of personalisation, and it assumes nothing. Keep the exact
    multiset of items the policy chose and shuffle which context receives which
    one. If the policy matches users to items they individually like, the
    shuffle destroys that and the value falls. If the value is unchanged, every
    point the policy scored came from WHICH items it selected, not from WHO got
    them -- the policy is a good item picker wearing a personalised policy's
    interface.

    This is a permutation test, so the null it reports is the policy's own
    choices under random assignment: no model, no fit, nothing to overfit. A
    collapsed policy returns a p-value of 1 by construction, which is the
    sanity check that the test is doing what it claims.

    The duration residual answers a different question -- whether the ITEMS are
    better than their length predicts. Both are reported because a policy can
    pass one and fail the other.
    """
    rng = np.random.default_rng(seed)
    index = np.arange(len(actions))
    observed = ~np.ma.getmaskarray(block)
    values = np.asarray(block.filled(0.0), dtype=np.float64)
    w = weights.astype(np.float64)

    def score(assignment: np.ndarray) -> float:
        keep = observed[index, assignment]
        total = w[keep].sum()
        if total == 0:
            return float("nan")
        return float((values[index, assignment][keep] * w[keep]).sum() / total)

    realised = score(actions)
    null = np.array([score(rng.permutation(actions)) for _ in range(n_permutations)])
    finite = null[np.isfinite(null)]
    if realised != realised or finite.size == 0:
        return {"realised": realised, "n_permutations": n_permutations, "usable": False}

    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    exceed = int((finite >= realised - 1e-12).sum())
    return {
        "realised": realised,
        "shuffled_mean": float(finite.mean()),
        "shuffled_std": std,
        "gain_from_matching": realised - float(finite.mean()),
        "z_score": (realised - float(finite.mean())) / std if std > 0 else float("nan"),
        "p_value": (exceed + 1) / (finite.size + 1),
        "n_permutations": int(finite.size),
        "usable": True,
    }


def duration_confound(
    train_actions: np.ndarray,
    train_raw: np.ndarray,
    per_action: np.ndarray,
    n_actions: int,
) -> dict | None:
    """Is the best constant policy simply the shortest video?

    KuaiRec's reward is ``1[watch_ratio >= 2.0]`` and ``watch_ratio =
    play_duration / video_duration``. A very short video clears that bar the
    moment it loops, so an item's value may be a property of its LENGTH rather
    than of anyone's preference. If the two rank together, the benchmark's
    ceiling is partly an artefact of the reward definition and the paper has to
    say so; if they do not, the objection is answered with a number.

    Returns ``None`` when the tabular feature is not a single column, since the
    question only arises for KuaiRec's one-dimensional state.
    """
    per_item = per_item_feature(train_actions, train_raw, n_actions)
    if per_item is None:
        return None

    usable = np.isfinite(per_item) & np.isfinite(per_action)
    if usable.sum() < 3:
        return None

    from scipy.stats import spearmanr

    rho, pvalue = spearmanr(per_item[usable], per_action[usable])
    ranked = np.argsort(np.nan_to_num(per_action, nan=-np.inf))[::-1]
    median = float(np.nanmedian(per_item[usable]))
    return {
        "feature": "video_duration (the only tabular column)",
        "n_items_compared": int(usable.sum()),
        "spearman_rho": float(rho),
        "spearman_p": float(pvalue),
        "median_duration_all_items": median,
        "top5_constants": [
            {
                "action": int(a),
                "exact_value": float(per_action[a]),
                "duration": float(per_item[a]),
                "duration_vs_median": float(per_item[a] / median) if median else float("nan"),
            }
            for a in ranked[:5]
        ],
    }


def policy_actions(policy, states: np.ndarray, n_actions: int) -> np.ndarray | None:
    """The action each context representative receives, or ``None`` if stochastic.

    ``UniformPolicy`` has no per-context action by construction and is reported
    as such rather than forced into an argmax it never makes.
    """
    dist = policy.action_distribution(states)
    if isinstance(dist, DeterministicPolicy):
        return np.asarray(dist.actions, dtype=np.int64)
    if isinstance(dist, UniformPolicy):
        return None
    dense = np.asarray(dist)
    if dense.ndim != 2:
        return None
    return np.asarray(dense.argmax(axis=1), dtype=np.int64)


def concentration(actions: np.ndarray, weights: np.ndarray, n_actions: int) -> dict:
    """How many distinct actions a policy uses, and how concentrated it is.

    ``n_distinct == 1`` is a collapsed policy: the state told it nothing, so it
    recommends one item to everyone. That is the difference between a weak
    learner and a policy class that cannot express personalisation.
    """
    counts = np.bincount(actions, weights=weights.astype(np.float64), minlength=n_actions)
    total = counts.sum()
    share = counts / total if total else counts
    nonzero = share[share > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    ranked = np.argsort(counts)[::-1]
    return {
        "n_distinct_actions": int((counts > 0).sum()),
        "n_contexts": int(len(actions)),
        "top_action": int(ranked[0]),
        "top_action_share": float(share[ranked[0]]),
        "entropy_nats": entropy,
        "max_entropy_nats": float(np.log(n_actions)),
        "collapsed": bool((counts > 0).sum() == 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--device", default="cpu",
        help="cpu by default: this is a diagnostic and must not compete for VRAM "
             "with a training job.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
    )

    config = load_config(args.config)
    if args.data_root is not None:
        config.paths.raw_data = Path(args.data_root).expanduser().resolve()
    config.paths.ensure_output_dirs()
    seed = args.seed if args.seed is not None else config.seed
    seed_everything(seed)

    prepared = _load_dataset(config, seed)
    test = prepared.arrays("test")
    true_rewards = _build_exact_ground_truth(config, prepared, test["user_index"], seed)
    if true_rewards is None:
        logger.error(
            "%s has no fully observed evaluation block, so there is no ground truth to "
            "take a ceiling from. This script is only meaningful on KuaiRec.",
            config.dataset.name,
        )
        return 1

    first_index, weights = context_representatives(test["user_index"])
    logger.info(
        "%d evaluation rounds carry %d distinct contexts (%.1f rounds each); every "
        "number below is computed on the %d representatives and weighted by round count.",
        len(test["user_index"]), len(first_index),
        len(test["user_index"]) / max(len(first_index), 1), len(first_index),
    )

    block = true_rewards[first_index].to_dense_masked()
    coverage = 1.0 - float(np.ma.getmaskarray(block).mean())
    logger.info(
        "Ground-truth block for these contexts: %d x %d cells, %.2f%% observed.",
        block.shape[0], block.shape[1], 100.0 * coverage,
    )

    report: dict = {
        "experiment": config.experiment_name,
        "dataset": config.dataset.name,
        "state_source": config.rl.state_source,
        "seed": seed,
        "n_contexts": int(len(first_index)),
        "n_rounds": int(len(test["user_index"])),
        "n_actions": int(prepared.n_actions),
        "block_coverage": coverage,
        "reference": None,
        "agents": {},
    }

    per_action, denominator, report["reference"] = reference_policies(block, weights)

    ref = report["reference"]
    logger.info(
        "ORACLE        %.5f   (best action per user; %d rounds scored)",
        ref["oracle"]["value"], ref["oracle"]["rounds_scored"],
    )
    logger.info(
        "BEST CONSTANT %.5f   (action %d for every user)",
        ref["best_constant"]["value"], ref["best_constant"]["action"],
    )

    from gcrl.phases.phase3_rl import build_state, load_policy

    train = prepared.arrays("train")
    report["reference"]["train_popularity"] = train_popularity_constants(
        train["actions"], train["rewards"], per_action, denominator, prepared.n_actions,
    )
    per_item = per_item_feature(train["actions"], train["raw_features"], prepared.n_actions)
    baseline = None if per_item is None else duration_baseline(per_item, per_action)

    confound = duration_confound(
        train["actions"], train["raw_features"], per_action, prepared.n_actions,
    )
    if confound is not None:
        report["reference"]["duration_confound"] = confound
        logger.info(
            "DURATION      Spearman rho = %+.3f (p = %.3g) between an item's duration and "
            "its value as a constant policy, over %d items.",
            confound["spearman_rho"], confound["spearman_p"], confound["n_items_compared"],
        )
        for row in confound["top5_constants"]:
            logger.info(
                "              constant %5d: exact %.5f, duration %.1f (%.2fx the median item)",
                row["action"], row["exact_value"], row["duration"], row["duration_vs_median"],
            )

    for name, pick in report["reference"]["train_popularity"].items():
        if not isinstance(pick, dict):
            continue
        logger.info(
            "POPULARITY    %.5f   (%s: action %d, rank %d of %d constants, %d log rows)",
            pick["exact_value"], name.replace("_", " "), pick["action"],
            pick["rank_among_constants"], prepared.n_actions, pick["log_interactions"],
        )
    train_state = build_state(config, train["raw_features"], train["user_index"], seed)
    test_state = build_state(config, test["raw_features"], test["user_index"], seed)
    states = test_state[first_index]
    device = resolve_device(args.device)
    suffix = f"_lambda{config.rl.cate_reward_weight:g}" if config.rl.cate_reward_weight else ""

    for agent in config.rl.agents:
        path = (
            Path(config.paths.policies)
            / f"{config.dataset.name}_{agent}_{config.rl.state_source}{suffix}_seed{seed}.pt"
        )
        if not any(c.exists() for c in (path, path.with_suffix(".npz"))):
            logger.warning("%s: no trained policy at %s -- skipped", agent, path.name)
            continue
        policy = load_policy(
            agent, path, prepared.n_actions, test_state.shape[1], config, seed, device,
            sample_state=train_state[:4], sample_actions=train["actions"][:4],
        )
        actions = policy_actions(policy, states, prepared.n_actions)
        if actions is None:
            report["agents"][agent] = {"stochastic": True, "note": "no per-context action"}
            logger.info("%-10s stochastic -- no per-context action to count", agent)
            continue

        stats = concentration(actions, weights, prepared.n_actions)
        # Fancy indexing a MaskedArray carries the mask across, so an
        # unobserved (context, chosen action) cell stays excluded rather than
        # silently becoming a zero inside a ground-truth column.
        chosen = block[np.arange(len(actions)), actions]
        value, scored, masked = weighted_masked_mean(chosen, weights)
        stats.update({
            "exact_value_recomputed": value,
            "rounds_scored": scored,
            "rounds_masked": masked,
            # Where this agent's single most-chosen action ranks among all 3,327
            # constant policies. A collapsed agent with a rank near 1 is not a
            # failed recommender -- it is a popularity detector, and the paper
            # has to say which.
            "top_action_rank_among_constants": constant_rank(per_action, stats["top_action"]),
        })
        personalisation = personalisation_test(block, weights, actions, seed=seed)
        stats["personalisation"] = personalisation
        if personalisation.get("usable"):
            logger.info(
                "%-10s   shuffling who gets what: %.5f -> %.5f (gain %+.5f, p = %.3f)%s",
                agent, personalisation["realised"], personalisation["shuffled_mean"],
                personalisation["gain_from_matching"], personalisation["p_value"],
                "   <-- MATCHING MATTERS" if personalisation["p_value"] < 0.05 else "",
            )

        if baseline is not None:
            adjusted = duration_adjusted_value(block, weights, baseline, actions)
            stats["duration_adjusted"] = adjusted
            logger.info(
                "%-10s   duration predicts %.5f of that; residual %+.5f%s",
                agent, adjusted["duration_predicted"],
                adjusted["duration_adjusted_residual"],
                "   <-- beats its own duration"
                if adjusted["duration_adjusted_residual"] > 0 else "",
            )
        report["agents"][agent] = stats
        logger.info(
            "%-10s exact %.5f | %5d distinct actions of %d | top action %.1f%% of contexts%s",
            agent, value, stats["n_distinct_actions"], prepared.n_actions,
            100.0 * stats["top_action_share"],
            f"  <-- COLLAPSED onto the #{stats['top_action_rank_among_constants']} constant"
            if stats["collapsed"] else "",
        )

    out = Path(config.paths.results) / f"reference_policies_{config.experiment_name}_seed{seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
