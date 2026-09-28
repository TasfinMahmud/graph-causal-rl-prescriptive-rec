"""Phase 2: heterogeneous treatment effect estimation.

The estimators are standard; what this phase adds is that it refuses to produce
numbers from a setup where no causal effect is identified. Every run passes
through :func:`gcrl.models.causal.validate_causal_setup` first, and a failure
raises rather than logging a warning that would scroll past.

Covariates come from the Phase 1 embeddings when
``causal.use_graph_embeddings`` is set. That switch is the honest form of the
"graph embeddings stabilise causal estimation" ablation: the same estimator,
the same treatment, the same outcome, with only the covariate representation
changed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig, resolve_device
from ..logging_utils import get_logger
from ..models.causal import (
    CausalDiagnostics,
    build_estimator,
    cate_summary,
    qini_coefficient,
    validate_causal_setup,
)

logger = get_logger(__name__)


@dataclass
class CausalEstimationResult:
    estimator: str
    dataset: str
    covariate_source: str
    n_train: int
    n_scored: int
    cate_mean: float
    cate_variance: float
    cate_std: float
    fraction_positive: float
    is_degenerate: bool
    qini: float | None
    fitting_seconds: float
    seed: int
    cate_path: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def prepare_covariates(
    config: ExperimentConfig,
    raw_features: np.ndarray,
    user_index: np.ndarray | None,
    seed: int,
) -> tuple:
    """Return ``(covariates, source_label)`` for the causal estimators."""
    if not config.causal.use_graph_embeddings:
        return np.nan_to_num(raw_features.astype(np.float64)), "raw_features"

    from .phase1_gnn import embedding_path, load_embeddings

    if user_index is None:
        raise ValueError("use_graph_embeddings=True requires a per-round user index")

    path = embedding_path(config, config.rl.gnn_architecture, seed)
    payload = load_embeddings(path)
    try:
        covariates = payload.for_users(user_index).astype(np.float64)
    except IndexError as exc:
        raise ValueError(f"{exc} (embedding file: {path.name})") from exc
    return covariates, f"gnn_embeddings:{config.rl.gnn_architecture}"


def estimate_cate(
    estimator_name: str,
    Y_train: np.ndarray,
    T_train: np.ndarray,
    X_train: np.ndarray,
    X_score: np.ndarray,
    config: ExperimentConfig,
    seed: int,
    device: str,
) -> tuple:
    """Fit one estimator on the training split and score every round.

    Fitting uses the training split only, so CATE values scored on the
    evaluation split are out-of-sample. The previous implementation fit on the
    first chunk and predicted across all 26M rows without saying so.
    """
    started = time.time()
    estimator = build_estimator(estimator_name, config.causal, seed=seed, device=device)

    logger.info(
        "[%s] fitting %s on %d units, %d covariates",
        config.dataset.name, estimator_name, len(Y_train), X_train.shape[1],
    )
    estimator.fit(Y_train, T_train, X=X_train)

    t0 = config.causal.treatment_reference
    t1 = config.causal.treatment_target
    if t0 is None or t1 is None:
        unique = np.unique(T_train)
        t0, t1 = int(unique.min()), int(unique.max())

    cate = np.asarray(estimator.effect(X_score, T0=t0, T1=t1), dtype=np.float64).ravel()

    if not np.isfinite(cate).all():
        n_bad = int((~np.isfinite(cate)).sum())
        raise RuntimeError(
            f"{estimator_name} produced {n_bad} non-finite CATE values. Refusing to report "
            f"them; investigate the covariates rather than replacing them with zeros."
        )
    return cate, time.time() - started


def run_phase2(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    train_mask: np.ndarray,
    config: ExperimentConfig,
    seed: int | None = None,
    propensities: np.ndarray | None = None,
    provenance: dict[str, list[str]] | None = None,
    covariate_source: str = "raw_features",
    feature_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Validate the setup, then fit every configured estimator.

    Returns:
        Mapping from estimator name to its per-round CATE vector.

    Raises:
        CausalValidationError: if the treatment/outcome setup identifies nothing.
    """
    seed = seed if seed is not None else config.seed
    device = resolve_device(config.device)

    logger.info("Phase 2: validating the causal setup before estimating anything")
    diagnostics: CausalDiagnostics = validate_causal_setup(
        Y, T, X,
        feature_names=feature_names,
        propensities=propensities,
        treatment_column=config.causal.treatment_column,
        outcome_column=config.causal.outcome_column,
        provenance=provenance,
        strict=True,
    )

    Y_train, T_train, X_train = Y[train_mask], T[train_mask], X[train_mask]
    results: list[CausalEstimationResult] = []
    cate_by_estimator: dict[str, np.ndarray] = {}

    cate_dir = Path(config.paths.cate)
    cate_dir.mkdir(parents=True, exist_ok=True)

    for estimator_name in config.causal.estimators:
        cate, seconds = estimate_cate(
            estimator_name, Y_train, T_train, X_train, X, config, seed, device
        )
        summary = cate_summary(cate)

        if summary["is_degenerate"]:
            logger.warning(
                "%s produced a constant CATE (variance %.3e). A degenerate estimator scores "
                "the best possible 'stability' while carrying no information, so this must be "
                "reported as a failure to fit, not as a low-variance result.",
                estimator_name, summary["variance"],
            )

        try:
            qini = qini_coefficient(cate[train_mask], Y_train, T_train)
        except (ValueError, ZeroDivisionError) as exc:
            logger.warning("Qini coefficient unavailable for %s: %s", estimator_name, exc)
            qini = None

        # Keyed by DATASET, not experiment_name: a CATE estimate depends on the data,
        # the estimators and the seed -- never on which ablation arm is running.
        # Keying by experiment_name made every arm recompute identical forests.
        from ..cli import cate_artefact_path

        cate_path = cate_artefact_path(config, estimator_name, seed)
        np.save(cate_path, cate)
        cate_by_estimator[estimator_name] = cate

        results.append(
            CausalEstimationResult(
                estimator=estimator_name,
                dataset=config.dataset.name,
                covariate_source=covariate_source,
                n_train=int(train_mask.sum()),
                n_scored=len(cate),
                cate_mean=summary["mean"],
                cate_variance=summary["variance"],
                cate_std=summary["std"],
                fraction_positive=summary["fraction_positive"],
                is_degenerate=bool(summary["is_degenerate"]),
                qini=qini,
                fitting_seconds=seconds,
                seed=seed,
                cate_path=str(cate_path),
            )
        )
        logger.info(
            "[%s] %s: CATE mean %.6g, variance %.6g, %.1f%% positive, qini %s (%.1fs)",
            config.dataset.name, estimator_name, summary["mean"], summary["variance"],
            100 * summary["fraction_positive"],
            f"{qini:.4f}" if qini is not None else "n/a", seconds,
        )

    output = Path(config.paths.results) / f"phase2_causal_{config.dataset.name}_seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "diagnostics": {
                    "n_units": diagnostics.n_units,
                    "treatment_balance": diagnostics.treatment_balance,
                    "outcome_treatment_correlation": diagnostics.outcome_treatment_correlation,
                    "overlap_violations": diagnostics.overlap_violations,
                    "warnings": diagnostics.warnings,
                },
                "covariate_source": covariate_source,
                "estimators": [r.as_row() for r in results],
            },
            handle, indent=2,
        )
    logger.info("Phase 2 results -> %s", output)
    return cate_by_estimator
