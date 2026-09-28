"""Typed configuration objects loaded from YAML.

Every tunable value lives in a config file and is validated on load. Nothing in
this project reads a hyperparameter from a literal buried in a function body,
which is what allowed the published configuration and the executed
configuration to drift apart previously.

The config that a run actually used is written next to that run's outputs, so
the paper's implementation-details section can be generated from what ran
rather than transcribed by hand.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing or internally inconsistent."""


@dataclass
class PathConfig:
    """Filesystem locations. All paths are resolved relative to ``root``."""

    root: Path = Path(".")
    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    embeddings: Path = Path("artifacts/embeddings")
    cate: Path = Path("artifacts/cate")
    policies: Path = Path("artifacts/policies")
    results: Path = Path("results")
    figures: Path = Path("results/figures")
    logs: Path = Path("logs")

    def resolve(self) -> PathConfig:
        root = Path(self.root).expanduser().resolve()
        resolved = {"root": root}
        for f in dataclasses.fields(self):
            if f.name == "root":
                continue
            value = Path(getattr(self, f.name))
            resolved[f.name] = value if value.is_absolute() else root / value
        return PathConfig(**resolved)

    def ensure_output_dirs(self) -> None:
        for name in ("embeddings", "cate", "policies", "results", "figures", "logs"):
            Path(getattr(self, name)).mkdir(parents=True, exist_ok=True)


@dataclass
class SplitConfig:
    """Train/validation/test partitioning.

    ``strategy`` is ``"temporal"`` (split on sorted timestamp, which prevents
    look-ahead leakage) or ``"random"``. Ratios must sum to 1.
    """

    strategy: str = "temporal"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    timestamp_column: str = "timestamp"

    def validate(self) -> None:
        if self.strategy not in {"temporal", "random"}:
            raise ConfigError(f"split.strategy must be 'temporal' or 'random', got {self.strategy!r}")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"split ratios must sum to 1.0, got {total:.6f}")
        for name in ("train_ratio", "val_ratio", "test_ratio"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"split.{name} must be > 0")


@dataclass
class GNNConfig:
    """Phase 1: graph representation learning.

    ``num_layers`` is the number of message-passing layers and is also the
    number of hops the neighbourhood sampler collects, so the receptive field
    always matches the model depth.
    """

    architectures: list[str] = field(
        default_factory=lambda: ["LightGCN", "GraphSAGE", "GAT", "NGCF", "GCN"]
    )
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    gat_heads: int = 4
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 50
    node_batch_size: int = 4096
    max_edges_per_batch: int | None = None
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-5
    num_negative_samples: int = 1

    def validate(self) -> None:
        if self.num_layers < 1:
            raise ConfigError("gnn.num_layers must be >= 1")
        if self.embedding_dim < 1 or self.hidden_dim < 1:
            raise ConfigError("gnn embedding_dim and hidden_dim must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("gnn.dropout must be in [0, 1)")
        if self.epochs < 1:
            raise ConfigError("gnn.epochs must be >= 1")
        if self.early_stopping_patience < 1:
            raise ConfigError("gnn.early_stopping_patience must be >= 1")


@dataclass
class CausalConfig:
    """Phase 2: heterogeneous treatment effect estimation.

    ``treatment_column`` and ``outcome_column`` are explicit and validated
    against each other, because a treatment that is a deterministic function of
    the outcome silently produces meaningless CATE estimates.
    """

    # Opt-in by default: a dataset must explicitly declare a valid treatment
    # before any causal quantity is estimated from it.
    enabled: bool = False
    estimators: list[str] = field(default_factory=lambda: ["XLearner", "SLearner", "GRF", "DragonNet"])
    primary_estimator: str = "XLearner"
    treatment_column: str = ""
    outcome_column: str = ""
    treatment_reference: int | None = None
    treatment_target: int | None = None
    use_graph_embeddings: bool = True
    use_logged_propensity: bool = True
    propensity_column: str = "propensity_score"
    propensity_clip: float | None = None
    n_estimators: int = 100
    max_depth: int = 10
    dragonnet_epochs: int = 50
    dragonnet_batch_size: int = 2048
    dragonnet_learning_rate: float = 1e-3
    dragonnet_alpha: float = 1.0

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.treatment_column:
            raise ConfigError("causal.treatment_column must be set when causal.enabled is true")
        if not self.outcome_column:
            raise ConfigError("causal.outcome_column must be set when causal.enabled is true")
        if self.treatment_column == self.outcome_column:
            raise ConfigError(
                "causal.treatment_column and causal.outcome_column are identical; "
                "the treatment effect of a variable on itself is not defined"
            )
        if self.primary_estimator not in self.estimators:
            raise ConfigError(
                f"causal.primary_estimator {self.primary_estimator!r} is not in causal.estimators"
            )
        if self.propensity_clip is not None and not 0 < self.propensity_clip < 0.5:
            raise ConfigError("causal.propensity_clip must be in (0, 0.5) when set")


@dataclass
class RLConfig:
    """Phase 3: offline policy optimisation.

    ``state_source`` selects what the agent observes: ``"gnn_embeddings"`` uses
    the Phase 1 output, ``"raw_features"`` uses the tabular columns. The two are
    the ablation contrast, so it is a config switch rather than two code paths.

    ``cate_reward_weight`` is the lambda in ``r = base_reward + lambda * CATE``.
    Setting it to 0.0 disables causal reward shaping and yields the ablation
    baseline for the same agent.
    """

    agents: list[str] = field(
        default_factory=lambda: ["DQN", "CQL", "IQL", "BCQ", "LinUCB", "NeuralUCB", "Random"]
    )
    state_source: str = "gnn_embeddings"
    gnn_architecture: str = "LightGCN"
    cate_reward_weight: float = 0.0
    cate_estimator: str = "XLearner"
    hidden_units: int = 256
    num_layers: int = 2
    learning_rate: float = 3e-4
    batch_size: int = 256
    epochs: int = 100
    gamma: float = 0.99
    target_update_interval: int = 1000
    bcq_action_flexibility: float = 0.3
    cql_alpha: float = 5.0
    iql_expectile: float = 0.7
    iql_beta: float = 3.0
    linucb_alpha: float = 1.0
    linucb_ridge_lambda: float = 1.0
    neuralucb_hidden: int = 100
    neuralucb_lambda: float = 1.0
    neuralucb_nu: float = 0.1
    neuralucb_epochs: int = 10

    def validate(self) -> None:
        if self.state_source not in {"gnn_embeddings", "raw_features"}:
            raise ConfigError(
                f"rl.state_source must be 'gnn_embeddings' or 'raw_features', got {self.state_source!r}"
            )
        if self.cate_reward_weight < 0:
            raise ConfigError("rl.cate_reward_weight must be >= 0")
        if not 0 < self.iql_expectile < 1:
            raise ConfigError("rl.iql_expectile must be in (0, 1)")
        if self.epochs < 1:
            raise ConfigError("rl.epochs must be >= 1")


@dataclass
class OPEConfig:
    """Phase 4: off-policy evaluation.

    ``estimators`` names the counterfactual estimators to report. ``exact``
    is only valid for a fully observed reward matrix, where the counterfactual
    reward is looked up rather than estimated.
    """

    estimators: list[str] = field(default_factory=lambda: ["snipw", "dr", "mrdr"])
    use_logged_propensity: bool = True
    propensity_column: str = "propensity_score"
    n_bootstrap: int = 100
    confidence_level: float = 0.95
    exact_matrix_path: str | None = None
    #: Number of folds used to CROSS-FIT the q-model behind DM, DR and MRDR.
    #: Fitting q on the same rounds it is then evaluated on lets the regressor
    #: memorise those rounds' noise, which inflates DM/DR/MRDR by an amount that
    #: differs per agent -- so it reorders the results table, not just its scale.
    #: Each fold's q-hat is fitted on the other folds only. 5 is the usual
    #: default; 2 is the minimum that leaves any out-of-fold rows at all.
    cross_fitting_folds: int = 5
    #: Where pi_b(a|x) comes from. ``"logged"`` requires the dataset to ship it.
    #: ``"estimated"`` fits a behaviour policy from the TRAINING log and records
    #: that the propensities are estimated, for datasets such as KuaiRec that
    #: log no propensity at all. There is deliberately no third option that
    #: substitutes a constant.
    behaviour_policy: str = "logged"
    #: Shrinkage of the per-user action distribution toward the marginal one,
    #: in pseudo-counts. Large values recover the marginal (popularity) policy;
    #: 0 would trust a single user's empirical frequencies completely.
    behaviour_policy_shrinkage: float = 10.0

    def validate(self) -> None:
        allowed = {"snipw", "ipw", "dm", "dr", "mrdr", "exact"}
        unknown = set(self.estimators) - allowed
        if unknown:
            raise ConfigError(f"unknown OPE estimators: {sorted(unknown)}; allowed: {sorted(allowed)}")
        if not self.estimators:
            raise ConfigError("ope.estimators must not be empty")
        if not 0 < self.confidence_level < 1:
            raise ConfigError("ope.confidence_level must be in (0, 1)")
        if self.cross_fitting_folds < 2:
            raise ConfigError(
                "ope.cross_fitting_folds must be >= 2; with one fold the reward model is "
                "fitted on the very rounds it scores, which is the in-sample bias this "
                "setting exists to remove"
            )
        if self.behaviour_policy not in {"logged", "estimated"}:
            raise ConfigError(
                f"ope.behaviour_policy must be 'logged' or 'estimated', got "
                f"{self.behaviour_policy!r}"
            )
        if self.behaviour_policy_shrinkage < 0:
            raise ConfigError("ope.behaviour_policy_shrinkage must be >= 0")


@dataclass
class DatasetConfig:
    """Dataset identity and the columns the pipeline reads."""

    name: str = ""
    loader: str = ""
    train_file: str = ""
    eval_file: str = ""
    user_column: str = "user_id"
    item_column: str = "item_id"
    reward_column: str = "reward"
    timestamp_column: str = "timestamp"
    feature_columns: list[str] = field(default_factory=list)
    feature_column_patterns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    num_actions: int | None = None
    subsample_rows: int | None = None
    #: Rows kept from ``eval_file`` as EVALUATION ROUNDS. The ground-truth
    #: reward block is always built from the whole evaluation file regardless,
    #: so subsampling rounds shortens Phase 4 without reducing what is known.
    eval_subsample_rows: int | None = None
    chunk_size: int = 500_000
    social_graph_file: str | None = None
    reward_definition: str | None = None

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("dataset.name must be set")
        if not self.loader:
            raise ConfigError("dataset.loader must be set")
        if self.subsample_rows is not None and self.subsample_rows < 1:
            raise ConfigError("dataset.subsample_rows must be >= 1 when set")
        if self.eval_subsample_rows is not None and self.eval_subsample_rows < 1:
            raise ConfigError("dataset.eval_subsample_rows must be >= 1 when set")
        if self.chunk_size < 1:
            raise ConfigError("dataset.chunk_size must be >= 1")


@dataclass
class ExperimentConfig:
    """Top-level configuration for one dataset's pipeline."""

    experiment_name: str = "unnamed"
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42])
    device: str = "auto"
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    ope: OPEConfig = field(default_factory=OPEConfig)

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ConfigError(f"device must be 'auto', 'cpu' or 'cuda[:N]', got {self.device!r}")
        if not self.seeds:
            raise ConfigError("seeds must contain at least one seed")
        for section in (self.dataset, self.split, self.gnn, self.causal, self.rl, self.ope):
            section.validate()

        if (
            self.rl.state_source == "gnn_embeddings"
            and self.rl.gnn_architecture not in self.gnn.architectures
        ):
            raise ConfigError(
                f"rl.gnn_architecture {self.rl.gnn_architecture!r} is not trained in "
                f"phase 1 (gnn.architectures = {self.gnn.architectures})"
            )
        if self.rl.cate_reward_weight > 0 and not self.causal.enabled:
            raise ConfigError(
                "rl.cate_reward_weight > 0 requires causal.enabled = true, otherwise "
                "there are no CATE estimates to shape the reward with"
            )

    def to_dict(self) -> dict[str, Any]:
        def _convert(value: Any) -> Any:
            if dataclasses.is_dataclass(value):
                return {f.name: _convert(getattr(value, f.name)) for f in dataclasses.fields(value)}
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, list):
                return [_convert(v) for v in value]
            return value

        return _convert(self)

    def save(self, path: Path) -> None:
        """Write the effective configuration next to a run's outputs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=False)


def _build_section(cls, payload: dict[str, Any] | None, section_name: str):
    if payload is None:
        return cls()
    if not isinstance(payload, dict):
        raise ConfigError(f"config section {section_name!r} must be a mapping")
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - valid
    if unknown:
        raise ConfigError(
            f"unknown keys in config section {section_name!r}: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )
    return cls(**payload)


def load_config(path: Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load, merge and validate an experiment configuration.

    A config may set ``extends: other.yaml`` to inherit from a sibling file;
    the child's keys are merged over the parent's, one level deep per section.

    Raises:
        ConfigError: if the file is missing, contains unknown keys, or fails
            any cross-section consistency check.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")

    parent_name = payload.pop("extends", None)
    if parent_name:
        parent_payload = load_raw(path.parent / parent_name)
        payload = _deep_merge(parent_payload, payload)

    if overrides:
        payload = _deep_merge(payload, overrides)

    known_sections = {"paths", "dataset", "split", "gnn", "causal", "rl", "ope"}
    scalars = {k: v for k, v in payload.items() if k not in known_sections}
    valid_scalars = {f.name for f in dataclasses.fields(ExperimentConfig)} - known_sections
    unknown = set(scalars) - valid_scalars
    if unknown:
        raise ConfigError(f"unknown top-level config keys: {sorted(unknown)}")

    config = ExperimentConfig(
        **scalars,
        paths=_build_section(PathConfig, payload.get("paths"), "paths"),
        dataset=_build_section(DatasetConfig, payload.get("dataset"), "dataset"),
        split=_build_section(SplitConfig, payload.get("split"), "split"),
        gnn=_build_section(GNNConfig, payload.get("gnn"), "gnn"),
        causal=_build_section(CausalConfig, payload.get("causal"), "causal"),
        rl=_build_section(RLConfig, payload.get("rl"), "rl"),
        ope=_build_section(OPEConfig, payload.get("ope"), "ope"),
    )
    config.paths = config.paths.resolve()
    config.validate()
    return config


def load_raw(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    parent_name = payload.pop("extends", None)
    if parent_name:
        payload = _deep_merge(load_raw(path.parent / parent_name), payload)
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``device`` to a concrete string, falling back to CPU.

    A CUDA device is always returned **with an explicit index** -- ``cuda:0``,
    never bare ``cuda``.

    That is not cosmetic. d3rlpy's ``torch_utility.map_location`` does::

        _, index = device.split(":")

    so a bare ``"cuda"`` raises ``ValueError: not enough values to unpack
    (expected 2, got 1)`` the moment a saved model is reloaded. Phase 3 trains
    and saves without touching that path, so the failure surfaces only in
    Phase 4 -- after the expensive work is done -- and its message names
    neither the device nor d3rlpy, which makes a bare ``"cuda"`` both late to
    fail and hard to attribute.

    Bare ``"cpu"`` is correct and is left alone; d3rlpy only splits on ``:``
    for CUDA devices. A request that already carries an index is passed
    through unchanged, so ``cuda:1`` keeps addressing GPU 1.
    """
    if requested != "auto":
        if requested.startswith("cuda"):
            try:
                import torch
            except ImportError:
                return "cpu"
            if not torch.cuda.is_available():
                logging_warning = (
                    "CUDA requested but unavailable; falling back to CPU. "
                    "Results remain valid but will be slower."
                )
                import logging

                logging.getLogger(__name__).warning(logging_warning)
                return "cpu"
        return _with_cuda_index(requested)

    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _with_cuda_index(device: str) -> str:
    """Give a bare ``cuda`` an explicit index; leave everything else alone."""
    return "cuda:0" if device == "cuda" else device
