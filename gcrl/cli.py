"""Command-line interface.

Every entry point takes ``--config`` and optional ``--seed``, so a run is fully
described by a config file plus a seed. There are no hardcoded paths and no
dependence on the working directory: the previous scripts used relative paths
like ``../data/...`` and broke unless invoked from one specific folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, load_config, resolve_device
from .logging_utils import configure_logging, get_logger
from .seeding import seed_everything

logger = get_logger(__name__)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="path to the experiment YAML")
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--data-root", type=Path, default=None, help="override paths.raw_data")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )


def _load(args: argparse.Namespace) -> tuple:
    config = load_config(args.config)
    if args.data_root is not None:
        config.paths.raw_data = Path(args.data_root).expanduser().resolve()
    config.paths.ensure_output_dirs()

    seed = args.seed if args.seed is not None else config.seed
    seed_everything(seed)

    import logging

    log_path = configure_logging(
        Path(config.paths.logs), f"{config.experiment_name}_seed{seed}",
        level=getattr(logging, args.log_level),
    )
    logger.info("Config: %s | seed: %d | device: %s", args.config, seed, resolve_device(config.device))
    if log_path:
        logger.info("Log file: %s", log_path)

    config.save(Path(config.paths.results) / f"effective_config_{config.experiment_name}_seed{seed}.yaml")
    return config, seed


def _fingerprint(payload: dict) -> str:
    """A short, stable digest of everything that produced an artefact.

    Every cache in this file keys on one of these. The rule that makes a cache
    safe is simple and absolute: if a field can change the artefact and is not
    in the payload, the cache will serve something built under different
    settings and every downstream number becomes quietly wrong.

    Keying Phase 1 and CATE artefacts on ``dataset.name`` alone is not
    sufficient. Because ``obd_fast.yaml`` inherits ``dataset.name: obd`` from
    ``obd.yaml``, a 300,000-row / 20-epoch run and a 2,000,000-row / 50-epoch
    run would resolve to the same filename, so whichever ran second would
    silently consume the first one's embeddings while reporting its own
    settings. That is the defect this function exists to prevent.
    """
    import hashlib
    import json

    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=10).hexdigest()


def _section(obj, exclude: frozenset[str] = frozenset()) -> dict:
    """Every field of a config section, as strings, minus an excluded set."""
    import dataclasses

    return {
        f.name: str(getattr(obj, f.name))
        for f in dataclasses.fields(obj)
        if f.name not in exclude
    }


def _data_identity(config: ExperimentConfig, seed: int) -> dict:
    """The inputs that determine which rows the pipeline is working with."""
    return {
        "dataset": _section(config.dataset),
        "split": _section(config.split),
        "raw_data": str(config.paths.raw_data),
        "seed": seed,
    }


def force_rebuild() -> bool:
    """Whether to ignore every artefact cache this run.

    Set ``GCRL_FORCE_REBUILD=1``. This replaces an earlier ``config.force_retrain``
    check that could never fire: ``ExperimentConfig`` has no such field, and the
    loader rejects unknown top-level keys, so there was no way to force a rebuild
    at all.
    """
    import os

    return bool(os.environ.get("GCRL_FORCE_REBUILD"))


def embedding_fingerprint(config: ExperimentConfig, architecture: str, seed: int) -> str:
    """Identity of a Phase 1 encoder artefact.

    Depends on the data, the architecture, and every GNN hyperparameter --
    ``embedding_dim``, ``epochs``, ``learning_rate`` and the rest all change the
    embeddings. ``architectures`` itself is excluded because it only lists which
    encoders a run trains; each artefact is for one named architecture.
    """
    return _fingerprint({
        **_data_identity(config, seed),
        "architecture": architecture,
        "gnn": _section(config.gnn, exclude=frozenset({"architectures"})),
        "schema": 2,
    })


def cate_fingerprint(config: ExperimentConfig, estimator: str, seed: int) -> str:
    """Identity of a Phase 2 CATE artefact.

    ``rl.gnn_architecture`` is included because Phase 2 *does* read it --
    ``phase2_causal.py`` selects its covariate embeddings by that name. An
    earlier docstring here claimed "Phase 2 does not read rl.* at all", which
    was simply false; an encoder ablation would have silently shared one CATE
    vector across architectures.
    """
    return _fingerprint({
        **_data_identity(config, seed),
        "estimator": estimator,
        "causal": _section(config.causal, exclude=frozenset({"estimators"})),
        "gnn_architecture": str(config.rl.gnn_architecture),
        "gnn": _section(config.gnn, exclude=frozenset({"architectures"})),
        "schema": 2,
    })


def cate_artefact_path(config: ExperimentConfig, estimator: str, seed: int) -> Path:
    """Where Phase 2 writes, and Phase 3 reads, one estimator's CATE vector.

    Keyed by a fingerprint of everything that produced it, so the three ablation
    arms -- which differ only in ``rl.state_source`` and ``rl.cate_reward_weight``,
    neither of which Phase 2 reads -- share one computation, while any change to
    the data, the estimator settings or the encoder produces a different file.

    Both writer and reader call this, so the two cannot drift apart.
    """
    stem = f"{config.dataset.name}_{estimator}_seed{seed}"
    return Path(config.paths.cate) / f"{stem}_{cate_fingerprint(config, estimator, seed)}_cate.npy"


def _dataset_cache_key(config: ExperimentConfig, seed: int) -> tuple[str, dict]:
    """A key covering every input that changes the prepared dataset.

    Beyond the dataset and split sections, this must include:

    * ``ope.use_logged_propensity`` and ``ope.estimators`` -- ``prepare_dataset``
      reads both to decide between *raising* and assigning a constant propensity
      of ``1/n_actions``. ``kuairec_validate_ope.yaml`` overrides only ``ope:``,
      so without these the key is identical to its parent's and a cache hit skips
      the guard entirely, running the paper's headline table against a fabricated
      uniform behaviour policy.
    * ``gnn.embedding_dim`` -- it sizes the random node features on the graph
      that ``prepare_dataset`` builds.

    The payload is written beside the entry so a cached dataset can be audited.
    """
    payload = {
        **_data_identity(config, seed),
        "ope": {
            "use_logged_propensity": str(config.ope.use_logged_propensity),
            "estimators": sorted(str(e) for e in config.ope.estimators),
            "propensity_column": str(config.ope.propensity_column),
            # These two select and parameterise the ESTIMATED behaviour policy,
            # so they change every value in __propensity. Omitting them would
            # let a run with a different behaviour-policy setting be served a
            # cached dataset carrying the other one's propensities.
            "behaviour_policy": str(config.ope.behaviour_policy),
            "behaviour_policy_shrinkage": str(config.ope.behaviour_policy_shrinkage),
        },
        "gnn_embedding_dim": str(config.gnn.embedding_dim),
        "schema": 3,
    }
    return _fingerprint(payload), payload


def _load_dataset(config: ExperimentConfig, seed: int):
    """Load the prepared dataset, from disk cache when one is valid.

    Each phase calls this independently, so a four-phase run over three
    variants and three seeds would otherwise read the source CSV 36 times.
    On OBD that source is 6.3 GB, which dominates total runtime.

    Set GCRL_NO_DATASET_CACHE=1 to bypass the cache entirely.
    """
    import os
    import pickle

    if os.environ.get("GCRL_NO_DATASET_CACHE") or force_rebuild():
        return _build_dataset(config, seed)

    key, payload = _dataset_cache_key(config, seed)
    cache_dir = Path(config.paths.processed_data) / "dataset_cache"
    entry = cache_dir / f"{key}.pkl"

    if entry.exists():
        try:
            prepared = pickle.loads(entry.read_bytes())
            logger.info("dataset cache hit (%s), source CSV not re-read", entry.name)
            return prepared
        except Exception as exc:
            # A corrupt or stale-format entry must never silently degrade a run.
            logger.warning("dataset cache unreadable (%s); rebuilding: %s", entry.name, exc)

    prepared = _build_dataset(config, seed)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Write to a temporary name and rename, so an interrupted write can
        # never leave a truncated entry that a later run might accept.
        tmp = entry.with_suffix(".pkl.tmp")
        tmp.write_bytes(pickle.dumps(prepared, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(entry)
        import json
        (cache_dir / f"{key}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("dataset cached to %s", entry)
    except Exception as exc:
        logger.warning("could not write dataset cache: %s", exc)
    return prepared


def _kuairec_matrix(filename: str, setting: str) -> str:
    """Which KuaiRec matrix a config filename names.

    Matched on the whole stem rather than by substring search, so a typo becomes
    an error instead of silently resolving to the other matrix -- which would
    train and evaluate on the same file again.
    """
    stem = Path(str(filename)).stem.lower()
    for matrix in ("big", "small"):
        if stem in {f"{matrix}_matrix", matrix}:
            return matrix
    raise ValueError(
        f"{setting}={filename!r} does not name a KuaiRec matrix. Expected "
        f"'big_matrix.csv' or 'small_matrix.csv'."
    )


def _build_exact_ground_truth(config: ExperimentConfig, prepared, test_user_index, seed: int):
    """The ground-truth reward table for the evaluation rounds, or ``None``.

    Returns ``None`` only when the dataset genuinely has no fully observed
    block. When ``exact`` is requested and this returns ``None``, Phase 4 raises
    rather than letting the estimator quietly disappear from the table.
    """
    if config.dataset.loader != "kuairec" or not config.dataset.eval_file:
        return None

    from .data.kuairec import build_exact_reward_lookup, load_kuairec

    eval_block = load_kuairec(
        Path(config.paths.raw_data) / "kuairec",
        matrix=_kuairec_matrix(config.dataset.eval_file, "dataset.eval_file"),
        subsample_rows=None, seed=seed,
        load_social_graph=False, load_user_features=False,
    )
    return build_exact_reward_lookup(
        eval_block,
        prepared.user_indexer,
        prepared.item_indexer,
        round_user_index=test_user_index,
        n_actions=prepared.n_actions,
        n_users=len(prepared.user_indexer),
    )


def _build_dataset(config: ExperimentConfig, seed: int):
    """Dispatch to the dataset loader named in the config."""
    from .pipeline import prepare_dataset

    if config.dataset.loader == "obd":
        from .data.obd import build_user_personas, load_obd

        policy, campaign = config.dataset.train_file.split("/")[:2]
        frame = load_obd(
            Path(config.paths.raw_data) / "obd",
            policy=policy, campaign=campaign,
            subsample_rows=config.dataset.subsample_rows, seed=seed,
        )
        persona_columns = [c for c in config.dataset.categorical_columns if c in frame.columns]
        if persona_columns:
            frame["user_persona"] = build_user_personas(frame, persona_columns)
            frame["user_persona"] = frame["user_persona"].astype("category").cat.codes
        else:
            frame["user_persona"] = 0

        return prepare_dataset(
            frame, config, user_column="user_persona", item_column="item_id",
            reward_column="click", propensity_column="propensity_score", seed=seed,
        )

    if config.dataset.loader == "kuairec":
        from .data.kuairec import KUAIREC_PROVENANCE, load_kuairec, load_kuairec_train_eval

        root = Path(config.paths.raw_data) / "kuairec"
        train_matrix = _kuairec_matrix(config.dataset.train_file, "dataset.train_file")

        if not config.dataset.eval_file:
            # One matrix, split temporally. Exact evaluation is NOT available in
            # this mode: nothing here is fully observed.
            data = load_kuairec(
                root, matrix=train_matrix,
                subsample_rows=config.dataset.subsample_rows, seed=seed,
            )
            return prepare_dataset(
                data.interactions, config, user_column="user_id", item_column="video_id",
                reward_column="reward", propensity_column=None,
                social_edges=data.social_edges, provenance=KUAIREC_PROVENANCE, seed=seed,
            )

        # The two-matrix protocol. dataset.eval_file was declared in the schema
        # and read by no code, so the protocol the docstrings described was not
        # the one that ran.
        bundle = load_kuairec_train_eval(
            root,
            train_matrix=train_matrix,
            eval_matrix=_kuairec_matrix(config.dataset.eval_file, "dataset.eval_file"),
            subsample_rows=config.dataset.subsample_rows,
            eval_subsample_rows=config.dataset.eval_subsample_rows,
            seed=seed,
        )
        return prepare_dataset(
            bundle.train.interactions, config, user_column="user_id", item_column="video_id",
            reward_column="reward", propensity_column=None,
            social_edges=bundle.train.social_edges, provenance=KUAIREC_PROVENANCE, seed=seed,
            eval_frame=bundle.eval_rounds.interactions,
            action_universe=bundle.action_universe,
            evaluation_block=config.dataset.eval_file,
        )

    raise ValueError(
        f"unknown dataset loader {config.dataset.loader!r}; available: obd, kuairec"
    )


def command_phase1(args: argparse.Namespace) -> int:
    from .phases.phase1_gnn import run_phase1
    from .pipeline import summarise_dataset

    config, seed = _load(args)
    prepared = _load_dataset(config, seed)

    summary = summarise_dataset(prepared, config)
    summary_path = Path(config.paths.results) / f"dataset_summary_{config.dataset.name}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Dataset summary -> %s\n%s", summary_path, json.dumps(summary, indent=2))

    run_phase1(prepared.graph, config, seed=seed)
    return 0


def command_phase2(args: argparse.Namespace) -> int:
    from .phases.phase2_causal import prepare_covariates, run_phase2

    config, seed = _load(args)
    if not config.causal.enabled:
        logger.error(
            "causal.enabled is false for %s, so Phase 2 has nothing to estimate. This is the "
            "correct setting for a dataset without a randomised intervention.", config.dataset.name,
        )
        return 1

    prepared = _load_dataset(config, seed)
    arrays = prepared.arrays("all")
    train_arrays = prepared.arrays("train")

    treatment = prepared.frame[config.causal.treatment_column].to_numpy()
    outcome = prepared.frame[config.causal.outcome_column].to_numpy(dtype=np.float64)

    reference = config.causal.treatment_reference
    if reference is not None:
        treatment = (treatment != reference).astype(np.int64)
        logger.info(
            "Binarised treatment %r against reference %r: %.2f%% treated",
            config.causal.treatment_column, reference, 100 * treatment.mean(),
        )

    covariates, source = prepare_covariates(
        config, arrays["raw_features"], arrays["user_index"], seed
    )
    train_mask = np.zeros(len(prepared.frame), dtype=bool)
    train_mask[: len(train_arrays["actions"])] = True

    run_phase2(
        Y=outcome, T=treatment, X=covariates, train_mask=train_mask, config=config, seed=seed,
        propensities=arrays["propensities"], provenance=prepared.provenance,
        covariate_source=source,
    )
    return 0


def command_phase3(args: argparse.Namespace) -> int:
    from .phases.phase3_rl import build_state, run_phase3, shape_rewards

    config, seed = _load(args)
    prepared = _load_dataset(config, seed)
    train = prepared.arrays("train")

    state = build_state(config, train["raw_features"], train["user_index"], seed)

    cate = None
    if config.rl.cate_reward_weight != 0.0:
        cate_path = cate_artefact_path(config, config.rl.cate_estimator, seed)
        if not cate_path.exists():
            logger.error("CATE estimates not found at %s; run Phase 2 first", cate_path)
            return 1
        cate = np.load(cate_path)
        # Phase 2 writes one CATE value per row of the FULL frame. Slicing it to
        # the training length silently succeeds when a stale vector happens to be
        # longer -- which is how a 2,000,000-row CATE got applied to a 300,000-row
        # run, passing every downstream length check. Assert instead.
        if len(cate) != len(prepared.frame):
            logger.error(
                "CATE at %s has %d entries but the prepared dataset has %d rows. "
                "This artefact was produced by a different run. Delete it and "
                "re-run Phase 2, or set GCRL_FORCE_REBUILD=1.",
                cate_path, len(cate), len(prepared.frame),
            )
            return 1
        cate = cate[: len(train["rewards"])]

    rewards = shape_rewards(train["rewards"], cate, config.rl.cate_reward_weight)
    run_phase3(state, train["actions"], rewards, prepared.n_actions, config, seed=seed)
    return 0


def command_phase4(args: argparse.Namespace) -> int:
    from .phases.phase3_rl import build_state, load_policy
    from .phases.phase4_ope import run_phase4

    config, seed = _load(args)
    prepared = _load_dataset(config, seed)
    train = prepared.arrays("train")
    test = prepared.arrays("test")

    train_state = build_state(config, train["raw_features"], train["user_index"], seed)
    test_state = build_state(config, test["raw_features"], test["user_index"], seed)

    device = resolve_device(config.device)
    suffix = f"_lambda{config.rl.cate_reward_weight:g}" if config.rl.cate_reward_weight else ""

    policies = {}
    for agent in config.rl.agents:
        path = (
            Path(config.paths.policies)
            / f"{config.dataset.name}_{agent}_{config.rl.state_source}{suffix}_seed{seed}.pt"
        )
        candidates = [path, path.with_suffix(".npz")]
        if not any(c.exists() for c in candidates):
            raise FileNotFoundError(
                f"trained policy for {agent!r} not found at {path}. Run Phase 3 first; the "
                f"agent is not skipped, because a missing policy previously became a "
                f"placeholder row in the results table."
            )
        policies[agent] = load_policy(
            agent, path, prepared.n_actions, test_state.shape[1], config, seed, device,
            sample_state=train_state[:4], sample_actions=train["actions"][:4],
        )

    true_rewards = _build_exact_ground_truth(config, prepared, test["user_index"], seed)
    if true_rewards is not None:
        # Every evaluation round is itself a recorded cell of the ground-truth
        # block, so the block must reproduce each round's own logged reward. An
        # unshared user or item indexer still yields a well-shaped matrix of
        # plausible numbers, and this is the only thing that would notice.
        true_rewards.assert_reproduces_logged_rewards(test["actions"], test["rewards"])

    run_phase4(
        policies, test_state, test["actions"], test["rewards"], test["propensities"],
        prepared.n_actions, config, seed=seed, true_rewards=true_rewards,
        propensity_source=prepared.propensity_source,
        evaluation_regime=prepared.evaluation_regime,
        round_context_id=test["user_index"],
        # A fully observed evaluation block is an ENUMERATION: its rounds are
        # one sample of users spread over many rows, not independent draws, so
        # the context is the independent unit. Declared here rather than
        # inferred, because this is the one place that knows it -- see
        # gcrl.phases.phase4_ope.resolve_independent_unit for what follows from
        # it, and why a plain log (OBD) answers the other way.
        independent_unit="context" if prepared.evaluation_block else "auto",
    )
    return 0


def command_run_all(args: argparse.Namespace) -> int:
    """Run every phase in order for one seed."""
    config = load_config(args.config)
    steps = [("phase1", command_phase1)]
    if config.causal.enabled:
        steps.append(("phase2", command_phase2))
    steps += [("phase3", command_phase3), ("phase4", command_phase4)]

    for name, function in steps:
        logger.info("=" * 70)
        logger.info("Running %s", name)
        logger.info("=" * 70)
        code = function(args)
        if code != 0:
            logger.error("%s exited with code %d; stopping", name, code)
            return code
    logger.info("All phases completed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gcrl",
        description="Graph-Enhanced Causal RL benchmark for prescriptive recommendation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, function, help_text in [
        ("phase1", command_phase1, "train GNN encoders and export node embeddings"),
        ("phase2", command_phase2, "estimate CATE (requires causal.enabled)"),
        ("phase3", command_phase3, "train offline RL agents"),
        ("phase4", command_phase4, "off-policy evaluation on the held-out test split"),
        ("run-all", command_run_all, "run every applicable phase in order"),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        _add_common_arguments(sub)
        sub.set_defaults(func=function)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
