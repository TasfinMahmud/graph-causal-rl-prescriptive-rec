"""DragonNet (Shi, Blei & Veitch, 2019).

This module was referenced by the previous pipeline but did not exist in the
repository, so ``from src.baselines_dragonnet import DragonNet`` raised
``ImportError`` and the DragonNet column of the results table could not have
been produced by the published code.

DragonNet learns a shared representation ``Phi(x)`` with three heads: a
propensity head ``g(x) = P(T=1 | x)`` and two outcome heads ``Q0``, ``Q1``. The
shared trunk is trained so that ``Phi(x)`` retains exactly the information
needed for treatment prediction, which is the sufficiency property that makes
it a valid adjustment set.

Targeted regularisation adds a single scalar parameter ``epsilon`` and a
perturbation term that gives the estimator a one-step correction. It is only a
correction if it is actually *applied*: the paper's estimate is read off the
perturbed outcome model

``Q~(x, t) = Q^(x, t) + eps^ * [ t / g^(x) - (1 - t) / (1 - g^(x)) ]``

so that ``tau(x) = Q~(x, 1) - Q~(x, 0) = Q1(x) - Q0(x) + eps^ / (g^(x)(1 - g^(x)))``.
:meth:`DragonNet.effect` returns that quantity. Fitting ``epsilon`` and then
discarding it, so that the raw ``Q1 - Q0`` is reported, would reduce the
targeted term to a regulariser on the trunk, and none of the doubly-robust
asymptotics would hold.

The correction is divided by ``g^(1 - g^)``, so it blows up exactly where
overlap fails. Propensities are therefore clipped to ``[c, 1 - c]``
(``propensity_clip``, default 0.01), which bounds the multiplier at
``1/c + 1/(1-c)`` ~= 101. That is a deviation from the paper, which assumes
strict overlap and does not clip. It is *not* silent: every clip is counted,
logged at WARNING, and left on the estimator as
``n_propensity_clipped_`` / ``propensity_range_`` so a caller can assert on it.
A run that reports clipping is reporting an overlap violation, and the targeted
correction there should be read as unreliable rather than as a refinement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 (conventional torch alias)

from ..logging_utils import get_logger

logger = get_logger(__name__)


def clip_propensity(propensity: torch.Tensor, clip: float) -> tuple[torch.Tensor, int]:
    """Clip ``propensity`` into ``[clip, 1 - clip]`` and report how often it bound.

    Returned alongside the clipped tensor is the number of entries that were
    actually moved. Callers surface that count rather than swallowing it: a
    propensity outside the bound means the targeted correction
    ``eps / (g (1 - g))`` was capped, so the estimate at that unit is no longer
    the quantity Shi et al. (2019) analyse.
    """
    if not 0.0 <= clip < 0.5:
        raise ValueError(f"propensity_clip must be in [0, 0.5), got {clip}")
    n_clipped = int(((propensity < clip) | (propensity > 1.0 - clip)).sum().item())
    return propensity.clamp(clip, 1.0 - clip), n_clipped


class _DragonNetModule(nn.Module):
    """Shared trunk with propensity and two outcome heads."""

    def __init__(self, input_dim: int, representation_dim: int = 200, outcome_dim: int = 100):
        super().__init__()
        self.representation = nn.Sequential(
            nn.Linear(input_dim, representation_dim), nn.ELU(),
            nn.Linear(representation_dim, representation_dim), nn.ELU(),
            nn.Linear(representation_dim, representation_dim), nn.ELU(),
        )
        # Propensity head is deliberately a single linear layer: keeping it
        # shallow forces the shared representation itself to carry the
        # treatment-predictive information, which is the point of the architecture.
        self.propensity_head = nn.Linear(representation_dim, 1)
        self.q0_head = nn.Sequential(
            nn.Linear(representation_dim, outcome_dim), nn.ELU(),
            nn.Linear(outcome_dim, outcome_dim), nn.ELU(),
            nn.Linear(outcome_dim, 1),
        )
        self.q1_head = nn.Sequential(
            nn.Linear(representation_dim, outcome_dim), nn.ELU(),
            nn.Linear(outcome_dim, outcome_dim), nn.ELU(),
            nn.Linear(outcome_dim, 1),
        )
        self.epsilon = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        phi = self.representation(x)
        return (
            self.q0_head(phi).squeeze(-1),
            self.q1_head(phi).squeeze(-1),
            self.propensity_head(phi).squeeze(-1),  # logits
        )


class DragonNet:
    """Estimator interface matching the EconML meta-learners used elsewhere.

    Args:
        alpha: Weight on the propensity (treatment prediction) loss.
        beta: Weight on the targeted regularisation term. Set to 0 to disable,
            which also disables the ``epsilon`` correction in :meth:`effect` --
            with ``beta = 0`` the parameter receives no gradient and stays at
            its initial zero, so the two are consistent either way.
        propensity_clip: Propensities are clipped into
            ``[clip, 1 - clip]`` before they appear in a denominator. Without
            this the targeted regularisation term diverges for near-deterministic
            treatment assignment.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        representation_dim: int = 200,
        outcome_dim: int = 100,
        epochs: int = 50,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        alpha: float = 1.0,
        beta: float = 1.0,
        propensity_clip: float = 0.01,
        device: str = "cpu",
        seed: int = 42,
        verbose: bool = True,
    ):
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        self.outcome_dim = outcome_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.propensity_clip = propensity_clip
        self.device = torch.device(device)
        self.seed = seed
        self.verbose = verbose
        self.model: _DragonNetModule | None = None
        self._outcome_mean = 0.0
        self._outcome_std = 1.0
        # Set by `effect`: how hard the clip bound bit on the last call.
        self.n_propensity_clipped_: int | None = None
        self.propensity_range_: tuple[float, float] | None = None

    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> DragonNet:
        """Fit on outcome ``Y``, binary treatment ``T`` and covariates ``X``.

        The argument order matches EconML's ``fit(Y, T, X=X)`` so the estimators
        are interchangeable at the call site.
        """
        Y = np.asarray(Y, dtype=np.float64).ravel()
        T = np.asarray(T).ravel()
        X = np.asarray(X, dtype=np.float32)

        unique_t = np.unique(T)
        if not np.all(np.isin(unique_t, [0, 1])):
            raise ValueError(
                f"DragonNet requires binary treatment coded 0/1, found values {unique_t[:10]}. "
                f"Binarise the treatment explicitly rather than letting it be cast silently."
            )
        if len(unique_t) < 2:
            raise ValueError(
                "treatment has no variation: every unit is in the same arm, so no "
                "treatment effect is identifiable"
            )
        if len(Y) != len(T) or len(Y) != len(X):
            raise ValueError(f"length mismatch: Y={len(Y)}, T={len(T)}, X={len(X)}")

        torch.manual_seed(self.seed)
        self.input_dim = X.shape[1]

        # Standardising the outcome keeps the three loss terms on comparable
        # scales; predictions are mapped back in `effect`.
        self._outcome_mean, self._outcome_std = float(Y.mean()), float(Y.std()) or 1.0
        y_scaled = (Y - self._outcome_mean) / self._outcome_std

        self.model = _DragonNetModule(
            self.input_dim, self.representation_dim, self.outcome_dim
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        x_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_scaled, dtype=torch.float32, device=self.device)
        t_t = torch.as_tensor(T, dtype=torch.float32, device=self.device)

        n = len(x_t)
        self.model.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(n, device=self.device)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = permutation[start : start + self.batch_size]
                bx, by, bt = x_t[idx], y_t[idx], t_t[idx]

                optimizer.zero_grad()
                q0, q1, logits = self.model(bx)

                y_pred = bt * q1 + (1.0 - bt) * q0
                outcome_loss = F.mse_loss(y_pred, by)
                treatment_loss = F.binary_cross_entropy_with_logits(logits, bt)
                loss = outcome_loss + self.alpha * treatment_loss

                if self.beta > 0:
                    propensity, _ = clip_propensity(
                        torch.sigmoid(logits), self.propensity_clip
                    )
                    # Clever covariate: T/g(x) - (1-T)/(1-g(x))
                    clever = bt / propensity - (1.0 - bt) / (1.0 - propensity)
                    y_perturbed = y_pred + self.model.epsilon * clever
                    loss = loss + self.beta * F.mse_loss(y_perturbed, by)

                loss.backward()
                optimizer.step()
                total += loss.item() * len(idx)

            if self.verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.info("DragonNet epoch %d/%d loss=%.6f", epoch + 1, self.epochs, total / n)

        return self

    def effect(self, X: np.ndarray, T0: int = 0, T1: int = 1) -> np.ndarray:
        """Return per-unit CATE on the original outcome scale.

        The estimate comes from the *perturbed* outcome model of Shi et al.
        (2019), not from the raw heads:

        ``tau(x) = Q1(x) - Q0(x) + eps^ * [1/g^(x) + 1/(1 - g^(x))]``

        which is ``Q~(x,1) - Q~(x,0)`` for the targeted-regularisation
        perturbation fitted in :meth:`fit`. Propensities are clipped with the
        same ``propensity_clip`` used during fitting, so the correction stays
        bounded where overlap is poor -- which is exactly where it is largest.

        Clipping is reported, not hidden: the number of clipped units is logged
        at WARNING and stored on ``self.n_propensity_clipped_``, with the
        unclipped range on ``self.propensity_range_``. A non-zero count means
        the estimate is capped rather than targeted at those units.
        """
        if self.model is None:
            raise RuntimeError("DragonNet must be fitted before calling effect()")
        if (T0, T1) != (0, 1):
            raise ValueError(
                f"DragonNet is defined for binary treatment; got T0={T0}, T1={T1}"
            )
        self.model.eval()
        x_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        out = []
        n_clipped, p_lo, p_hi = 0, float("inf"), float("-inf")
        with torch.no_grad():
            for start in range(0, len(x_t), 8192):
                q0, q1, logits = self.model(x_t[start : start + 8192])
                tau = q1 - q0
                if self.beta > 0:
                    raw = torch.sigmoid(logits)
                    p_lo, p_hi = min(p_lo, float(raw.min())), max(p_hi, float(raw.max()))
                    propensity, batch_clipped = clip_propensity(raw, self.propensity_clip)
                    n_clipped += batch_clipped
                    # Q~(x,1) - Q~(x,0) = (Q1 - Q0) + eps * [1/g + 1/(1-g)].
                    tau = tau + self.model.epsilon * (
                        1.0 / propensity + 1.0 / (1.0 - propensity)
                    )
                out.append((tau * self._outcome_std).cpu().numpy())

        if self.beta > 0 and len(x_t):
            self.n_propensity_clipped_ = n_clipped
            self.propensity_range_ = (p_lo, p_hi)
            if n_clipped:
                logger.warning(
                    "DragonNet.effect: %d/%d propensities fell outside [%.3g, %.3g] and were "
                    "clipped (unclipped range [%.4g, %.4g]). The targeted correction is capped "
                    "at those units, so their CATE is not the Shi et al. estimand -- this is an "
                    "overlap violation, not a refinement.",
                    n_clipped, len(x_t), self.propensity_clip, 1.0 - self.propensity_clip,
                    p_lo, p_hi,
                )
        return np.concatenate(out) if out else np.empty(0)

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        """Return the fitted ``P(T=1 | x)``."""
        if self.model is None:
            raise RuntimeError("DragonNet must be fitted before predicting propensity")
        self.model.eval()
        x_t = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        with torch.no_grad():
            return torch.sigmoid(self.model(x_t)[2]).cpu().numpy()

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("nothing to save: DragonNet has not been fitted")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "input_dim": self.input_dim,
             "outcome_mean": self._outcome_mean, "outcome_std": self._outcome_std,
             "representation_dim": self.representation_dim, "outcome_dim": self.outcome_dim,
             # Without these two, a reloaded estimator would apply a different
             # epsilon correction from the one that was fitted.
             "beta": self.beta, "propensity_clip": self.propensity_clip},
            path,
        )

    def load(self, path: Path) -> DragonNet:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = state["input_dim"]
        self.representation_dim = state["representation_dim"]
        self.outcome_dim = state["outcome_dim"]
        self.model = _DragonNetModule(
            self.input_dim, self.representation_dim, self.outcome_dim
        ).to(self.device)
        self.model.load_state_dict(state["state_dict"])
        self._outcome_mean = state["outcome_mean"]
        self._outcome_std = state["outcome_std"]
        self.beta = state.get("beta", self.beta)
        self.propensity_clip = state.get("propensity_clip", self.propensity_clip)
        return self
