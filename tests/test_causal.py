"""Tests for causal validity checking and CATE estimation.

The headline test is :class:`TestCircularTreatmentDetection`, which reproduces
the exact setup that previously produced meaningless CATE values and confirms
the pipeline now refuses it.
"""

import logging

import numpy as np
import pytest
import torch

from gcrl.config import CausalConfig
from gcrl.models.causal import (
    CausalValidationError,
    build_estimator,
    cate_summary,
    check_treatment_provenance,
    detect_functional_dependence,
    qini_coefficient,
    validate_causal_setup,
)


@pytest.fixture
def kuairec_circular_setup():
    """Reproduces T = 1[watch_ratio > 1] with Y = play_duration.

    Since watch_ratio = play_duration / video_duration, the treatment is a
    deterministic function of the outcome.
    """
    rng = np.random.default_rng(0)
    n = 3000
    video_duration = rng.uniform(5, 60, n)
    play_duration = rng.uniform(0, 120, n)
    treatment = ((play_duration / video_duration) > 1.0).astype(int)
    covariates = np.column_stack([video_duration, rng.normal(size=n)])
    return play_duration, treatment, covariates


@pytest.fixture
def valid_randomised_setup():
    rng = np.random.default_rng(1)
    n = 3000
    covariates = np.column_stack([rng.uniform(5, 60, n), rng.normal(size=n)])
    treatment = (rng.random(n) < 0.5).astype(int)
    outcome = covariates[:, 1] + treatment * (1 + 0.01 * covariates[:, 0]) + 0.3 * rng.normal(size=n)
    return outcome, treatment, covariates


KUAIREC_PROVENANCE = {
    "watch_ratio": ["play_duration", "video_duration"],
    "treated": ["watch_ratio"],
}


class TestProvenance:
    def test_detects_treatment_derived_from_outcome(self):
        problems = check_treatment_provenance("treated", "play_duration", KUAIREC_PROVENANCE)
        assert problems
        assert "watch_ratio" in problems[0]

    def test_reports_the_full_dependency_path(self):
        problems = check_treatment_provenance("treated", "play_duration", KUAIREC_PROVENANCE)
        assert "treated -> watch_ratio -> play_duration" in problems[0]

    def test_accepts_an_independent_treatment(self):
        assert check_treatment_provenance("item_id", "click", {"click": ["logged_response"]}) == []

    def test_rejects_treatment_equal_to_outcome(self):
        assert check_treatment_provenance("y", "y", {}) != []

    def test_terminates_on_cyclic_provenance(self):
        """A malformed lineage must not hang the validator."""
        assert check_treatment_provenance("a", "z", {"a": ["b"], "b": ["a"]}) == []


class TestCircularTreatmentDetection:
    def test_declared_provenance_blocks_the_run(self, kuairec_circular_setup):
        Y, T, X = kuairec_circular_setup
        with pytest.raises(CausalValidationError, match="derived from the outcome"):
            validate_causal_setup(
                Y, T, X, treatment_column="treated", outcome_column="play_duration",
                provenance=KUAIREC_PROVENANCE,
            )

    def test_statistical_backstop_fires_without_declared_provenance(self, kuairec_circular_setup):
        """Catches the same bug when lineage was never written down."""
        Y, T, X = kuairec_circular_setup
        accuracy = detect_functional_dependence(Y, T, X)
        assert accuracy is not None and accuracy > 0.99

    def test_backstop_does_not_fire_on_a_valid_setup(self, valid_randomised_setup):
        Y, T, X = valid_randomised_setup
        assert detect_functional_dependence(Y, T, X) is None

    def test_valid_setup_passes(self, valid_randomised_setup):
        Y, T, X = valid_randomised_setup
        diagnostics = validate_causal_setup(Y, T, X, propensities=np.full(len(Y), 0.5))
        assert diagnostics.n_units == len(Y)
        assert 0.4 < diagnostics.treatment_balance < 0.6


class TestOtherValidityChecks:
    def test_rejects_treatment_with_no_variation(self):
        rng = np.random.default_rng(0)
        with pytest.raises(CausalValidationError, match="single value"):
            validate_causal_setup(rng.random(100), np.ones(100, dtype=int), rng.random((100, 2)))

    def test_rejects_covariate_that_determines_treatment(self):
        rng = np.random.default_rng(0)
        T = (rng.random(500) < 0.5).astype(int)
        X = np.column_stack([T.astype(float), rng.normal(size=500)])
        with pytest.raises(CausalValidationError, match="within-stratum variation"):
            validate_causal_setup(rng.random(500), T, X, feature_names=["leak", "ok"])

    def test_flags_overlap_violations(self, valid_randomised_setup):
        Y, T, X = valid_randomised_setup
        propensities = np.full(len(Y), 0.5)
        propensities[:10] = 1e-6
        diagnostics = validate_causal_setup(Y, T, X, propensities=propensities)
        assert diagnostics.overlap_violations == 10
        assert any("propensity" in w for w in diagnostics.warnings)

    def test_non_strict_mode_logs_instead_of_raising(self, kuairec_circular_setup):
        Y, T, X = kuairec_circular_setup
        diagnostics = validate_causal_setup(
            Y, T, X, treatment_column="treated", outcome_column="play_duration",
            provenance=KUAIREC_PROVENANCE, strict=False,
        )
        assert diagnostics is not None

    def test_length_mismatch_raises(self):
        with pytest.raises(CausalValidationError, match="length mismatch"):
            validate_causal_setup(np.zeros(10), np.zeros(5), np.zeros((10, 2)))


class TestCateSummary:
    def test_flags_a_degenerate_constant_estimate(self):
        """A constant CATE scores perfect 'stability' while carrying no signal."""
        assert cate_summary(np.full(100, 2.5))["is_degenerate"] is True

    def test_does_not_flag_a_varying_estimate(self):
        assert cate_summary(np.random.default_rng(0).normal(size=100))["is_degenerate"] is False

    def test_reports_the_expected_fields(self):
        summary = cate_summary(np.array([-1.0, 0.0, 1.0, 2.0]))
        assert summary["mean"] == pytest.approx(0.5)
        assert summary["fraction_positive"] == pytest.approx(0.5)
        assert summary["n"] == 4

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cate_summary(np.array([]))


class TestQini:
    def test_informative_ranking_beats_random(self, valid_randomised_setup):
        Y, T, X = valid_randomised_setup
        informative = qini_coefficient(1 + 0.01 * X[:, 0], Y, T)
        random_ranking = qini_coefficient(np.random.default_rng(3).normal(size=len(Y)), Y, T)
        assert informative > random_ranking

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            qini_coefficient(np.zeros(5), np.zeros(4), np.zeros(5))


class TestEstimatorConstruction:
    @pytest.mark.parametrize("name", ["XLearner", "SLearner", "GRF", "DragonNet"])
    def test_every_configured_estimator_can_be_built(self, name):
        estimator = build_estimator(name, CausalConfig(n_estimators=5, max_depth=3), seed=0)
        assert hasattr(estimator, "fit") and hasattr(estimator, "effect")

    def test_unknown_estimator_raises_rather_than_substituting(self):
        with pytest.raises(ValueError, match="unknown causal estimator"):
            build_estimator("NotAnEstimator", CausalConfig(), seed=0)


class TestDragonNet:
    def test_recovers_a_known_heterogeneous_effect(self):
        from gcrl.models.dragonnet import DragonNet

        rng = np.random.default_rng(0)
        n = 3000
        X = rng.normal(size=(n, 5))
        T = (rng.random(n) < 0.5).astype(int)
        true_cate = 2.0 + X[:, 0]
        Y = X[:, 1] + T * true_cate + 0.1 * rng.normal(size=n)

        model = DragonNet(
            epochs=40, batch_size=256, representation_dim=64, outcome_dim=32, verbose=False
        ).fit(Y, T, X)
        estimated = model.effect(X)
        assert np.corrcoef(estimated, true_cate)[0, 1] > 0.9
        assert estimated.mean() == pytest.approx(true_cate.mean(), abs=0.3)

    def test_rejects_non_binary_treatment(self):
        from gcrl.models.dragonnet import DragonNet

        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="binary treatment"):
            DragonNet(epochs=1, verbose=False).fit(
                rng.random(60), np.tile([0, 1, 2], 20), rng.random((60, 3))
            )

    def test_effect_before_fit_raises(self):
        from gcrl.models.dragonnet import DragonNet

        with pytest.raises(RuntimeError, match="must be fitted"):
            DragonNet().effect(np.zeros((5, 3)))


def _shi_et_al_cate(model, X):
    """Shi, Blei & Veitch (2019), sec. 3.2, computed independently of ``effect``.

    ``Q~(x, t) = Q^(x, t) + eps^ * [ t / g^(x) - (1 - t) / (1 - g^(x)) ]``

    evaluated at ``t = 1`` and ``t = 0`` and differenced, then mapped back onto
    the original outcome scale. Only the fitted network parameters are borrowed
    from the estimator; the perturbation is applied here from the paper's
    equation rather than read back out of the implementation.
    """
    model.model.eval()
    with torch.no_grad():
        q0, q1, logits = model.model(torch.as_tensor(np.asarray(X, dtype=np.float32)))
    q0 = q0.numpy().astype(np.float64)
    q1 = q1.numpy().astype(np.float64)
    g = 1.0 / (1.0 + np.exp(-logits.numpy().astype(np.float64)))
    g = np.clip(g, model.propensity_clip, 1.0 - model.propensity_clip)
    epsilon = float(model.model.epsilon.item())

    q_tilde_1 = q1 + epsilon * (1.0 / g)
    q_tilde_0 = q0 + epsilon * (-1.0 / (1.0 - g))
    return (q_tilde_1 - q_tilde_0) * model._outcome_std


def _raw_head_difference(model, X):
    """``(Q1 - Q0)`` on the original scale: what the pre-fix ``effect`` returned."""
    model.model.eval()
    with torch.no_grad():
        q0, q1, _ = model.model(torch.as_tensor(np.asarray(X, dtype=np.float32)))
    return (q1 - q0).numpy().astype(np.float64) * model._outcome_std


def _synthetic(seed=0, n=600, overlap_strength=0.7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    propensity = 1.0 / (1.0 + np.exp(-overlap_strength * X[:, 0]))
    T = (rng.random(n) < propensity).astype(int)
    true_cate = 1.0 - 0.2 * X[:, 1]
    Y = X[:, 2] + T * true_cate + 0.3 * rng.normal(size=n)
    return Y, T, X, true_cate


class TestDragonNetTargetedRegularisation:
    """DEFECT 3: epsilon was fitted and then thrown away.

    The pre-fix ``effect`` returned the raw ``Q1 - Q0``, which makes the
    targeted term a plain regulariser on the trunk and forfeits the
    doubly-robust asymptotics the docstring claims.
    """

    @pytest.fixture(scope="class")
    def fitted(self):
        from gcrl.models.dragonnet import DragonNet

        Y, T, X, _ = _synthetic()
        model = DragonNet(
            epochs=15, batch_size=256, representation_dim=32, outcome_dim=16,
            verbose=False, seed=0,
        ).fit(Y, T, X)
        return model, X

    def test_effect_equals_the_perturbed_outcome_model(self, fitted):
        model, X = fitted
        np.testing.assert_allclose(model.effect(X), _shi_et_al_cate(model, X),
                                   rtol=1e-4, atol=1e-6)

    def test_epsilon_is_applied_not_merely_fitted(self, fitted):
        """Setting epsilon must move the estimate, which fails if ``effect``
        never reads the parameter.

        The shift is pure algebra --
        ``tau(eps2) - tau(eps1) = (eps2 - eps1) * [1/g + 1/(1-g)] * sigma_Y`` --
        so it is checked exactly rather than approximately.
        """
        model, X = fitted
        original = float(model.model.epsilon.item())
        try:
            with torch.no_grad():
                model.model.epsilon.fill_(0.0)
            at_zero = model.effect(X)
            with torch.no_grad():
                model.model.epsilon.fill_(0.5)
            at_half = model.effect(X)
        finally:
            with torch.no_grad():
                model.model.epsilon.fill_(original)

        g = np.clip(model.predict_propensity(X).astype(np.float64),
                    model.propensity_clip, 1.0 - model.propensity_clip)
        expected_shift = 0.5 * (1.0 / g + 1.0 / (1.0 - g)) * model._outcome_std
        np.testing.assert_allclose(at_half - at_zero, expected_shift, rtol=1e-4, atol=1e-5)

        # 1/g + 1/(1-g) >= 4 for any g, so the shift cannot be lost in noise.
        assert np.abs(at_half - at_zero).min() > 1.9 * model._outcome_std

    def test_epsilon_zero_recovers_the_raw_head_difference(self, fitted):
        """Sanity anchor for the test above: at eps = 0 the correction vanishes
        and the estimate coincides with the pre-fix quantity."""
        model, X = fitted
        original = float(model.model.epsilon.item())
        try:
            with torch.no_grad():
                model.model.epsilon.fill_(0.0)
            np.testing.assert_allclose(model.effect(X), _raw_head_difference(model, X),
                                       rtol=1e-5, atol=1e-7)
        finally:
            with torch.no_grad():
                model.model.epsilon.fill_(original)

    def test_fitted_epsilon_actually_moves_the_estimate(self, fitted):
        """With beta > 0 the fitted epsilon is non-zero, so the corrected and
        uncorrected estimates genuinely differ on real fitted weights."""
        model, X = fitted
        assert abs(float(model.model.epsilon.item())) > 1e-6, "epsilon never left its init"
        difference = np.abs(model.effect(X) - _raw_head_difference(model, X))
        assert difference.mean() > 1e-4

    def test_beta_zero_disables_the_correction(self):
        from gcrl.models.dragonnet import DragonNet

        Y, T, X, _ = _synthetic(seed=1, n=400)
        model = DragonNet(
            epochs=5, batch_size=256, representation_dim=32, outcome_dim=16,
            beta=0.0, verbose=False, seed=1,
        ).fit(Y, T, X)
        assert float(model.model.epsilon.item()) == 0.0
        np.testing.assert_allclose(model.effect(X), _raw_head_difference(model, X),
                                   rtol=1e-5, atol=1e-7)


class TestDragonNetPropensityClipping:
    """Clipping bounds the correction where overlap fails; it must not be silent."""

    def test_clip_propensity_bounds_and_counts(self):
        from gcrl.models.dragonnet import clip_propensity

        raw = torch.tensor([0.001, 0.4, 0.5, 0.6, 0.999])
        clipped, n_clipped = clip_propensity(raw, 0.01)
        assert n_clipped == 2
        torch.testing.assert_close(clipped, torch.tensor([0.01, 0.4, 0.5, 0.6, 0.99]))

    def test_clip_propensity_counts_nothing_when_inside_the_bound(self):
        from gcrl.models.dragonnet import clip_propensity

        raw = torch.tensor([0.2, 0.5, 0.8])
        clipped, n_clipped = clip_propensity(raw, 0.01)
        assert n_clipped == 0
        torch.testing.assert_close(clipped, raw)

    def test_clip_propensity_rejects_a_nonsensical_bound(self):
        from gcrl.models.dragonnet import clip_propensity

        with pytest.raises(ValueError, match="propensity_clip must be in"):
            clip_propensity(torch.tensor([0.5]), 0.7)

    def test_good_overlap_reports_no_clipping(self):
        from gcrl.models.dragonnet import DragonNet

        Y, T, X, _ = _synthetic(seed=2, n=400, overlap_strength=0.7)
        model = DragonNet(
            epochs=10, batch_size=256, representation_dim=32, outcome_dim=16,
            verbose=False, seed=2,
        ).fit(Y, T, X)
        model.effect(X)
        assert model.n_propensity_clipped_ == 0

    def test_boundary_propensities_are_counted_and_logged(self, caplog):
        """The clip that keeps the correction finite has to be reported: at a
        clipped unit the returned CATE is capped rather than targeted.

        The propensity head is forced to a boundary logit so the condition is
        exercised deterministically rather than hoping a fit drifts there.
        """
        from gcrl.models.dragonnet import DragonNet

        Y, T, X, _ = _synthetic(seed=0, n=200)
        model = DragonNet(
            epochs=3, batch_size=256, representation_dim=32, outcome_dim=16,
            verbose=False, seed=0,
        ).fit(Y, T, X)
        with torch.no_grad():
            model.model.propensity_head.weight.zero_()
            model.model.propensity_head.bias.fill_(8.0)  # sigmoid(8) ~= 0.99966 > 0.99

        with caplog.at_level(logging.WARNING, logger="gcrl.models.dragonnet"):
            estimated = model.effect(X)

        assert model.n_propensity_clipped_ == len(X)
        assert model.propensity_range_[1] > 0.99
        assert np.isfinite(estimated).all()
        assert any("clipped" in record.message for record in caplog.records), (
            "clipping happened but nothing was logged"
        )

    def test_a_poor_overlap_fit_reports_clipping(self):
        """End-to-end counterpart: a near-deterministic assignment really does
        drive g^ past the bound during an ordinary fit."""
        from gcrl.models.dragonnet import DragonNet

        Y, T, X, _ = _synthetic(seed=0, n=800, overlap_strength=9.0)
        model = DragonNet(
            epochs=30, batch_size=256, representation_dim=32, outcome_dim=16,
            verbose=False, seed=0,
        ).fit(Y, T, X)
        model.effect(X)
        assert model.n_propensity_clipped_ > 0
        assert model.propensity_range_[0] < model.propensity_clip

    def test_correction_stays_finite_under_a_boundary_propensity(self):
        """Even with g^ pinned at the bound the estimate is finite, because the
        multiplier is capped at 1/c + 1/(1-c)."""
        from gcrl.models.dragonnet import DragonNet

        Y, T, X, _ = _synthetic(seed=1, n=400, overlap_strength=12.0)
        model = DragonNet(
            epochs=15, batch_size=256, representation_dim=32, outcome_dim=16,
            verbose=False, seed=1,
        ).fit(Y, T, X)
        estimated = model.effect(X)
        assert np.isfinite(estimated).all()
        cap = (1.0 / model.propensity_clip + 1.0 / (1.0 - model.propensity_clip))
        bound = np.abs(_raw_head_difference(model, X)) + abs(
            float(model.model.epsilon.item())
        ) * cap * model._outcome_std
        assert np.all(np.abs(estimated) <= bound + 1e-5)
