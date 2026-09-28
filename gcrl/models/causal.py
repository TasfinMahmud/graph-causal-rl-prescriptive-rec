"""Heterogeneous treatment effect estimators and their validity checks.

The estimators themselves are thin wrappers around EconML, so the substance of
this module is the validation in :func:`validate_causal_setup`.

That validation exists to catch a specific and easily overlooked failure.
Consider a KuaiRec specification with treatment ``T = watch_ratio > 1.0`` and
outcome ``Y = play_duration``, where ``watch_ratio = play_duration /
video_duration``. The treatment is then a deterministic function of the
outcome. No causal quantity is identified under that specification, yet every
estimator will still return numbers that are large and plausible looking. A
CATE pipeline that does not check for this cannot distinguish a real effect
from an algebraic identity, so the check is run before any estimator is
fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)


class CausalValidationError(ValueError):
    """Raised when the treatment/outcome/covariate setup cannot identify an effect."""


@dataclass
class CausalDiagnostics:
    """Assumption checks reported alongside every CATE estimate."""

    n_units: int
    treatment_balance: float
    outcome_treatment_correlation: float
    propensity_min: float | None
    propensity_max: float | None
    overlap_violations: int
    warnings: list[str]

    def summary(self) -> str:
        lines = [
            f"units={self.n_units:,}",
            f"treated fraction={self.treatment_balance:.4f}",
            f"corr(T, Y)={self.outcome_treatment_correlation:+.4f}",
        ]
        if self.propensity_min is not None:
            lines.append(f"propensity range=[{self.propensity_min:.4f}, {self.propensity_max:.4f}]")
            lines.append(f"overlap violations={self.overlap_violations:,}")
        return " | ".join(lines)


def check_treatment_provenance(
    treatment_column: str,
    outcome_column: str,
    provenance: dict[str, list[str]] | None = None,
) -> list[str]:
    """Check whether the treatment is derived from the outcome, by declaration.

    ``provenance`` maps a derived column to the columns it was computed from,
    e.g. ``{"watch_ratio": ["play_duration", "video_duration"]}``. The check
    walks the treatment's dependency closure and reports any path that reaches
    the outcome column.

    This is the primary guard against a circular setup, and it is deterministic:
    unlike a statistical test it cannot miss a dependence that happens to be
    non-linear, and it cannot fire spuriously on a genuinely strong effect. Its
    cost is that provenance must be declared -- which is itself worth doing,
    because writing down where each variable comes from is how this class of
    error gets noticed.
    """
    provenance = provenance or {}
    problems: list[str] = []

    if treatment_column == outcome_column:
        return [f"treatment and outcome are the same column ({treatment_column!r})"]

    seen: set = set()
    stack: list[tuple] = [(treatment_column, [treatment_column])]
    while stack:
        column, path = stack.pop()
        if column in seen:
            continue
        seen.add(column)
        for parent in provenance.get(column, []):
            new_path = path + [parent]
            if parent == outcome_column:
                problems.append(
                    "the treatment is derived from the outcome: "
                    + " -> ".join(new_path)
                    + f". A treatment computed from {outcome_column!r} carries no "
                    "interventional meaning, so no causal effect is identified."
                )
            else:
                stack.append((parent, new_path))
    return problems


def detect_functional_dependence(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    threshold: float = 0.995,
    max_rows: int = 20_000,
    seed: int = 42,
) -> float | None:
    """Statistical backstop for an undeclared circular treatment.

    Fits a depth-3 decision tree to predict ``T`` from ``Y``, ``X`` and the
    pairwise ratios and differences between them, scored by cross-validation.
    A shallow tree reaching near-perfect accuracy on such features means a
    simple deterministic rule recovers the treatment -- the signature of a
    treatment defined from the outcome.

    Ratios matter specifically: ``T = (Y / X_j > c)`` is invisible to a linear
    correlation check and hard for an axis-aligned tree on raw columns, but
    trivial once ``Y / X_j`` is available as a feature.

    Returns:
        The cross-validated accuracy when it exceeds ``threshold``, else ``None``.
    """
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:  # pragma: no cover
        logger.warning("scikit-learn unavailable; skipping functional-dependence check")
        return None

    Y = np.asarray(Y, dtype=np.float64).ravel()
    T = np.asarray(T).ravel()
    X = np.asarray(X, dtype=np.float64)

    if len(np.unique(T)) < 2:
        return None

    if len(Y) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Y), max_rows, replace=False)
        Y, T, X = Y[idx], T[idx], X[idx]

    epsilon = 1e-9
    features = [Y.reshape(-1, 1), X]
    for column in range(X.shape[1]):
        features.append((Y / (X[:, column] + epsilon)).reshape(-1, 1))
        features.append((Y - X[:, column]).reshape(-1, 1))
    expanded = np.nan_to_num(np.hstack(features), posinf=0.0, neginf=0.0)

    accuracy = float(
        cross_val_score(
            DecisionTreeClassifier(max_depth=3, random_state=seed), expanded, T, cv=3
        ).mean()
    )
    return accuracy if accuracy >= threshold else None


def validate_causal_setup(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    feature_names: list[str] | None = None,
    propensities: np.ndarray | None = None,
    treatment_column: str = "treatment",
    outcome_column: str = "outcome",
    provenance: dict[str, list[str]] | None = None,
    correlation_threshold: float = 0.99,
    dependence_threshold: float = 0.995,
    overlap_epsilon: float = 1e-3,
    strict: bool = True,
) -> CausalDiagnostics:
    """Check the assumptions a CATE estimate depends on.

    Checks performed, in order of reliability:

    1. **Provenance** -- is the treatment derived from the outcome? Deterministic,
       from the declared column lineage. Always an error when it fires.
    2. **Treatment variation** -- both arms must be populated.
    3. **Covariate/treatment collinearity** -- a covariate that reproduces ``T``
       leaves no within-stratum variation to identify an effect from.
    4. **Functional dependence** -- statistical backstop for an undeclared
       circular treatment. Raised as a warning for human review, not an error,
       because a genuinely large treatment effect also makes ``T`` predictable
       from ``Y``.
    5. **Overlap / positivity** -- propensities bounded away from 0 and 1.

    Args:
        strict: When ``True``, identification failures raise. When ``False`` they
            are logged. Use ``False`` only for exploration, never for results.

    Raises:
        CausalValidationError: if ``strict`` and an identification check fails.
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    T = np.asarray(T).ravel()
    X = np.asarray(X, dtype=np.float64)
    problems: list[str] = []
    warnings_found: list[str] = []

    if not (len(Y) == len(T) == len(X)):
        raise CausalValidationError(f"length mismatch: Y={len(Y)}, T={len(T)}, X={len(X)}")

    problems.extend(check_treatment_provenance(treatment_column, outcome_column, provenance))

    unique_t = np.unique(T)
    if len(unique_t) < 2:
        problems.append(
            f"treatment takes a single value ({unique_t.tolist()}); no contrast exists"
        )
    treated_fraction = float(np.mean(unique_t.max() == T)) if len(unique_t) else 0.0
    if 0 < treated_fraction < 0.01 or 0.99 < treated_fraction < 1:
        warnings_found.append(
            f"severe treatment imbalance: {treated_fraction:.4%} of units in the treated arm"
        )

    ty_corr = 0.0
    if Y.std() > 0 and T.astype(np.float64).std() > 0:
        ty_corr = float(np.corrcoef(T.astype(np.float64), Y)[0, 1])
        if abs(ty_corr) >= correlation_threshold:
            problems.append(
                f"|corr(T, Y)| = {abs(ty_corr):.4f} >= {correlation_threshold}; the treatment "
                f"is very nearly the outcome"
            )

    for column in range(X.shape[1]):
        if X[:, column].std() == 0 or T.astype(np.float64).std() == 0:
            continue
        corr = abs(float(np.corrcoef(X[:, column], T.astype(np.float64))[0, 1]))
        if corr >= correlation_threshold:
            name = feature_names[column] if feature_names else f"column {column}"
            problems.append(
                f"covariate {name!r} has |corr| = {corr:.4f} with the treatment; it either "
                f"is the treatment or determines it, leaving no within-stratum variation"
            )

    dependence = detect_functional_dependence(Y, T, X, threshold=dependence_threshold)
    if dependence is not None:
        warnings_found.append(
            f"a depth-3 decision tree recovers the treatment from the outcome and covariates "
            f"with {dependence:.4%} cross-validated accuracy. This is the signature of a "
            f"treatment defined as a function of the outcome. Declare the provenance of "
            f"{treatment_column!r} and confirm it is not derived from {outcome_column!r}."
        )

    p_min = p_max = None
    overlap_violations = 0
    if propensities is not None:
        propensities = np.asarray(propensities, dtype=np.float64).ravel()
        p_min, p_max = float(propensities.min()), float(propensities.max())
        overlap_violations = int(
            np.sum((propensities < overlap_epsilon) | (propensities > 1 - overlap_epsilon))
        )
        if overlap_violations:
            warnings_found.append(
                f"{overlap_violations:,} units have propensity outside "
                f"[{overlap_epsilon}, {1 - overlap_epsilon}]; inverse weights for these units "
                f"dominate any IPW-family estimate"
            )

    diagnostics = CausalDiagnostics(
        n_units=len(Y),
        treatment_balance=treated_fraction,
        outcome_treatment_correlation=ty_corr,
        propensity_min=p_min,
        propensity_max=p_max,
        overlap_violations=overlap_violations,
        warnings=warnings_found,
    )

    for message in warnings_found:
        logger.warning("Causal diagnostic: %s", message)

    if problems:
        detail = "\n  - ".join(problems)
        if strict:
            raise CausalValidationError(
                f"the causal setup does not identify a treatment effect:\n  - {detail}"
            )
        for message in problems:
            logger.error("Causal validation failure (strict=False): %s", message)

    logger.info("Causal diagnostics: %s", diagnostics.summary())
    return diagnostics


def build_estimator(name: str, config, seed: int = 42, device: str = "cpu") -> Any:
    """Instantiate a CATE estimator by name.

    All estimators expose ``fit(Y, T, X=X)`` and ``effect(X)``.

    Raises:
        ValueError: for an unknown estimator name.
        ImportError: with an actionable message if a dependency is missing --
            never a silent substitution of a different estimator.
    """
    known = {"XLearner", "SLearner", "GRF", "DragonNet"}
    if name not in known:
        raise ValueError(
            f"unknown causal estimator {name!r}. Available: {', '.join(sorted(known))}"
        )

    if name == "DragonNet":
        from .dragonnet import DragonNet

        return DragonNet(
            epochs=config.dragonnet_epochs,
            batch_size=config.dragonnet_batch_size,
            learning_rate=config.dragonnet_learning_rate,
            alpha=config.dragonnet_alpha,
            device=device,
            seed=seed,
        )

    try:
        from econml.dml import CausalForestDML
        from econml.metalearners import SLearner, XLearner
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"estimator {name!r} requires econml. Install it with "
            f"`pip install econml`; it is listed in requirements.txt."
        ) from exc

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression

    def _regressor():
        return RandomForestRegressor(
            n_estimators=config.n_estimators, max_depth=config.max_depth,
            random_state=seed, n_jobs=-1,
        )

    def _classifier():
        return RandomForestClassifier(
            n_estimators=config.n_estimators, max_depth=config.max_depth,
            random_state=seed, n_jobs=-1,
        )

    if name == "XLearner":
        return XLearner(models=_regressor(), propensity_model=LogisticRegression(max_iter=1000))
    if name == "SLearner":
        return SLearner(overall_model=_regressor())
    if name == "GRF":
        return CausalForestDML(
            model_y=_regressor(), model_t=_classifier(), discrete_treatment=True,
            n_estimators=config.n_estimators, max_depth=config.max_depth, random_state=seed,
        )

    raise AssertionError(f"estimator {name!r} passed the name check but was not constructed")


def cate_summary(cate: np.ndarray) -> dict[str, float]:
    """Descriptive statistics for a CATE vector.

    Variance alone is a poor headline metric: a degenerate estimator that
    predicts one constant for every unit achieves a variance of exactly zero and
    would appear to be the most 'stable' model in the comparison. ``is_degenerate``
    makes that case visible rather than rewarding it.
    """
    cate = np.asarray(cate, dtype=np.float64).ravel()
    if cate.size == 0:
        raise ValueError("cannot summarise an empty CATE vector")

    variance = float(np.var(cate))
    return {
        "n": int(cate.size),
        "mean": float(np.mean(cate)),
        "variance": variance,
        "std": float(np.std(cate)),
        "min": float(np.min(cate)),
        "max": float(np.max(cate)),
        "q25": float(np.quantile(cate, 0.25)),
        "median": float(np.median(cate)),
        "q75": float(np.quantile(cate, 0.75)),
        "fraction_positive": float(np.mean(cate > 0)),
        "is_degenerate": bool(variance < 1e-12),
    }


def qini_coefficient(cate: np.ndarray, outcomes: np.ndarray, treatments: np.ndarray) -> float:
    """Qini coefficient: how well the CATE ranking targets responsive units.

    A policy-relevant complement to CATE variance. Units are ordered by
    predicted effect and the cumulative incremental outcome is compared against
    random targeting; the normalised area between the curves is returned.
    Positive means the ranking beats random targeting, zero means it does not.
    """
    cate = np.asarray(cate, dtype=np.float64).ravel()
    outcomes = np.asarray(outcomes, dtype=np.float64).ravel()
    treatments = np.asarray(treatments).ravel()
    if not (len(cate) == len(outcomes) == len(treatments)):
        raise ValueError("cate, outcomes and treatments must have equal length")

    order = np.argsort(-cate)
    y, t = outcomes[order], (treatments[order] == np.max(treatments)).astype(np.float64)

    treated_cumulative = np.cumsum(y * t)
    control_cumulative = np.cumsum(y * (1 - t))
    n_treated = np.cumsum(t)
    n_control = np.cumsum(1 - t)

    with np.errstate(divide="ignore", invalid="ignore"):
        uplift = treated_cumulative - control_cumulative * np.divide(
            n_treated, n_control, out=np.zeros_like(n_treated), where=n_control > 0
        )
    uplift = np.nan_to_num(uplift)

    n = len(cate)
    random_line = np.linspace(0, uplift[-1], n)
    denominator = abs(uplift[-1]) * n / 2.0
    if denominator < 1e-12:
        return 0.0
    return float(np.sum(uplift - random_line) / denominator)
