"""Offline RL agents and contextual bandit baselines.

Every agent here implements the algorithm it is named after. Where the
published algorithm has no discrete-action implementation available in
``d3rlpy`` -- IQL is the case -- it is implemented here from its paper rather
than substituted with a different algorithm under the same name.

A note on what changed and why. The previous implementation reported results
under three names that did not match the code: ``IQL`` was ``DiscreteSAC``,
``LinUCB`` was per-arm ridge regression with no confidence bound, and
``NeuralUCB`` was a plain MLP regressor. Since the defining feature of a UCB
algorithm is the exploration bonus, and the defining feature of IQL is
expectile regression, those were different algorithms. All three are
implemented properly below.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 (conventional torch alias)

from ..evaluation.ope import DENSE_ACTION_DIST_CELL_BUDGET
from ..logging_utils import get_logger

logger = get_logger(__name__)


class BasePolicy(ABC):
    """Common interface for every agent, so Phase 4 evaluates them identically."""

    name: str = "base"
    requires_action_distribution: bool = True
    #: Size of the discrete action space. Every subclass sets this in __init__;
    #: action_distribution() needs it to shape the returned array.
    n_actions: int = 0

    @abstractmethod
    def fit(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Return the greedy action per row."""

    def action_distribution(self, observations: np.ndarray):
        """Return ``pi_e(a | x)``.

        Deterministic policies return a :class:`DeterministicPolicy`, which the
        estimators consume without ever materialising the dense
        ``(n_rows, n_actions)`` matrix. That matrix is 8.6 GB for a 100k-round
        split over KuaiRec's 10,728 actions, and is almost entirely zeros.
        """
        from ..evaluation.ope import DeterministicPolicy

        return DeterministicPolicy(self.predict(observations), self.n_actions)

    def distribution_is_dense(self, n_rounds: int) -> bool:
        """Whether :meth:`action_distribution` materialises a dense matrix.

        Phase 4 asks this BEFORE calling :meth:`action_distribution`, so an
        agent that cannot answer sparsely at this scale is refused in the first
        second instead of inside a NumPy allocation. The default is False
        because the default implementation returns a
        :class:`~gcrl.evaluation.ope.DeterministicPolicy`; an agent that may
        return a dense array must override this and agree with itself.
        """
        return False

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        ...


class RandomPolicy(BasePolicy):
    """Uniform random action selection -- the reference baseline.

    This is a real, evaluated policy. Its value must be measured on the same
    test split with the same estimator as every other agent, never asserted as
    a constant.
    """

    name = "Random"

    def __init__(self, n_actions: int, seed: int = 42):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def fit(self, observations, actions, rewards) -> None:
        return None

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self.rng.integers(0, self.n_actions, size=len(observations))

    def action_distribution(self, observations: np.ndarray):
        """Uniform, which is the policy's true distribution -- not a one-hot sample."""
        from ..evaluation.ope import UniformPolicy

        return UniformPolicy(len(observations), self.n_actions)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as handle:
            pickle.dump({"n_actions": self.n_actions, "seed": self.seed}, handle)

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        self.n_actions = state["n_actions"]
        self.rng = np.random.default_rng(state["seed"])


class LinUCB(BasePolicy):
    """Disjoint LinUCB (Li et al., 2010).

    Maintains per-arm ``A_a = lambda I + sum x x^T`` and ``b_a = sum r x``, with
    ``theta_a = A_a^-1 b_a`` and the upper confidence bound

    ``ucb_a(x) = theta_a^T x + alpha * sqrt(x^T A_a^-1 x)``

    The second term is the confidence width and is what makes this LinUCB
    rather than ridge regression; omitting it removes the algorithm's entire
    exploration mechanism.

    Sufficient statistics accumulate across calls to :meth:`fit`, so streaming
    the data in chunks is equivalent to fitting on all of it at once. The
    previous implementation called ``Ridge.fit`` per chunk, which discarded
    every chunk but the last.
    """

    name = "LinUCB"

    def __init__(self, n_actions: int, context_dim: int, alpha: float = 1.0, ridge_lambda: float = 1.0):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.alpha = alpha
        self.ridge_lambda = ridge_lambda
        self.A = np.repeat(
            (ridge_lambda * np.eye(context_dim, dtype=np.float64))[None, :, :], n_actions, axis=0
        )
        self.b = np.zeros((n_actions, context_dim), dtype=np.float64)
        self.n_observed = np.zeros(n_actions, dtype=np.int64)
        self._theta: np.ndarray | None = None
        self._inv_flat: np.ndarray | None = None
        self._cache_valid = False

    def fit(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        observations = np.asarray(observations, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float64)

        if observations.shape[1] != self.context_dim:
            raise ValueError(
                f"context dimension mismatch: model expects {self.context_dim}, "
                f"got {observations.shape[1]}"
            )
        if actions.max(initial=-1) >= self.n_actions or actions.min(initial=0) < 0:
            raise ValueError(
                f"actions must lie in [0, {self.n_actions}); observed range "
                f"[{actions.min(initial=0)}, {actions.max(initial=-1)}]"
            )

        for action in np.unique(actions):
            mask = actions == action
            context = observations[mask]
            self.A[action] += context.T @ context
            self.b[action] += context.T @ rewards[mask]
            self.n_observed[action] += int(mask.sum())

        self._cache_valid = False

    def _refresh_cache(self) -> None:
        """Invert every arm's design matrix once and cache the result.

        ``A`` changes only in :meth:`fit`, so the inverses are computed there
        rather than on every prediction call.
        """
        inverses = np.linalg.inv(self.A)
        self._theta = np.einsum("aij,aj->ai", inverses, self.b)
        # Flattened inverses let the confidence width be computed as one matmul
        # (see predict), which is orders of magnitude faster than looping arms.
        self._inv_flat = inverses.reshape(self.n_actions, -1)
        self._cache_valid = True

    def predict(self, observations: np.ndarray, row_chunk: int = 2048) -> np.ndarray:
        """Greedy action under the upper confidence bound.

        The confidence width ``x^T A_a^-1 x`` is needed for every (row, arm)
        pair. Looping over arms is quadratic in the catalogue size and becomes
        unusable at KuaiRec's 10,728 actions -- roughly 19 seconds for 200 rows.

        Instead, note that

            x^T M x = sum_{j,k} M[j,k] * (x_j * x_k) = <vec(M), vec(x x^T)>

        so stacking ``vec(x x^T)`` per row and ``vec(A_a^-1)`` per arm turns the
        whole computation into a single ``(n, d^2) @ (d^2, n_actions)`` matmul
        that BLAS handles directly. Rows are processed in chunks to bound the
        peak memory of the intermediate.
        """
        observations = np.asarray(observations, dtype=np.float64)
        if observations.shape[1] != self.context_dim:
            raise ValueError(
                f"context dimension mismatch: model expects {self.context_dim}, "
                f"got {observations.shape[1]}"
            )
        if not self._cache_valid:
            self._refresh_cache()
        assert self._theta is not None and self._inv_flat is not None  # set by _refresh_cache

        out = np.empty(len(observations), dtype=np.int64)
        for start in range(0, len(observations), row_chunk):
            block = observations[start : start + row_chunk]
            mean = block @ self._theta.T

            if self.alpha == 0.0:
                out[start : start + len(block)] = np.argmax(mean, axis=1)
                continue

            # vec(x x^T) for every row in the block -> (n_block, d^2)
            outer = (block[:, :, None] * block[:, None, :]).reshape(len(block), -1)
            widths = outer @ self._inv_flat.T
            scores = mean + self.alpha * np.sqrt(np.maximum(widths, 0.0))
            out[start : start + len(block)] = np.argmax(scores, axis=1)
        return out

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            Path(path).with_suffix(".npz"),
            A=self.A, b=self.b, n_observed=self.n_observed,
            alpha=self.alpha, ridge_lambda=self.ridge_lambda,
        )

    def load(self, path: Path) -> None:
        state = np.load(Path(path).with_suffix(".npz"))
        self.A, self.b, self.n_observed = state["A"], state["b"], state["n_observed"]
        self.alpha, self.ridge_lambda = float(state["alpha"]), float(state["ridge_lambda"])
        self.n_actions, self.context_dim = self.b.shape
        self._cache_valid = False


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        layers, dim = [], in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(dim, hidden), nn.ReLU()]
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuralUCB(BasePolicy):
    """NeuralUCB (Zhou et al., 2020) with the standard diagonal approximation.

    The reward for each arm is modelled by a shared network ``f(x; theta)``. The
    exploration bonus uses the gradient of the network at the candidate arm,

    ``ucb_a(x) = f_a(x; theta) + nu * sqrt(g_a(x)^T Z^-1 g_a(x))``

    where ``Z`` accumulates ``g g^T``. The full matrix is quadratic in the
    parameter count, so as in the paper's practical variant ``Z`` is kept
    diagonal. Without this bonus the algorithm is an MLP regressor, not
    NeuralUCB.
    """

    name = "NeuralUCB"

    #: Working-set ceiling for the vectorised confidence-width computation. It
    #: sets the row and action block sizes only; results do not depend on it.
    ucb_memory_budget: int = 256 * 1024**2

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        hidden: int = 100,
        lambda_: float = 1.0,
        nu: float = 0.1,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.device = torch.device(device)
        self.network = _MLP(context_dim, hidden, n_actions).to(self.device)
        self.lambda_ = lambda_
        self.nu = nu
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        n_params = sum(p.numel() for p in self.network.parameters())
        self.Z_diag = torch.full((n_params,), lambda_, dtype=torch.float64, device=self.device)

    def fit(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        """Train the reward network and accumulate the gradient covariance."""
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rew = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.network.train()
        n = len(obs)
        for epoch in range(self.epochs):
            permutation = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                predicted = self.network(obs[idx]).gather(1, act[idx].unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(predicted, rew[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            logger.debug("NeuralUCB epoch %d/%d loss=%.6f", epoch + 1, self.epochs, epoch_loss / n)

        self._accumulate_gradient_covariance(obs, act)

    def _accumulate_gradient_covariance(self, obs: torch.Tensor, act: torch.Tensor, cap: int = 2000) -> None:
        """Update the diagonal of ``Z`` with per-round gradient outer products.

        Capped at ``cap`` rounds because this requires one backward pass per
        round; the diagonal estimate is stable well before the cap.
        """
        self.network.eval()
        n = min(len(obs), cap)
        for i in range(n):
            self.network.zero_grad()
            self.network(obs[i : i + 1])[0, act[i]].backward()
            grad = torch.cat(
                [p.grad.flatten() for p in self.network.parameters() if p.grad is not None]
            ).double()
            self.Z_diag += grad**2
        self.network.zero_grad()

    # ------------------------------------------------------------------
    # Exploration bonus.
    #
    # The bonus is ``nu * sqrt(g_a(x)^T Z^-1 g_a(x))`` with
    # ``g_a(x) = grad_theta f_a(x; theta)`` taken at *one* context ``x``: there
    # is one gradient per (row, action) pair. Backpropagating a batched sum --
    # ``self.network(obs)[:, a].sum().backward()`` -- yields ``sum_i g_a(x_i)``
    # instead, which is a different vector and is identical for every row of the
    # batch, so it collapses the bonus to a single per-action constant. That is
    # not NeuralUCB and must not be used here.
    # ------------------------------------------------------------------

    def _parameter_slices(self) -> list[tuple[torch.nn.Parameter, int, int]]:
        """``(parameter, start, stop)`` in the flat layout ``Z_diag`` uses.

        ``Z_diag`` is accumulated from ``torch.cat`` over
        ``self.network.parameters()``, so the same iteration order defines the
        slice of ``Z`` belonging to each parameter tensor.
        """
        out, offset = [], 0
        for p in self.network.parameters():
            out.append((p, offset, offset + p.numel()))
            offset += p.numel()
        return out

    def _linear_trunk_and_head(self):
        """Split the network into a feature trunk and a per-action linear head.

        Returns ``(trunk, head, linear_indices)`` when the network is a
        ``Sequential`` ending in ``nn.Linear`` and every trunk parameter belongs
        to an ``nn.Linear`` inside the trunk; otherwise ``(None, None, None)``,
        which sends :meth:`_ucb_widths` down the exact row-by-row path.
        """
        seq = getattr(self.network, "net", None)
        if not isinstance(seq, nn.Sequential) or len(seq) < 2:
            return None, None, None
        head = seq[-1]
        if not isinstance(head, nn.Linear):
            return None, None, None
        trunk = seq[:-1]
        indices, covered = [], set()
        for i, mod in enumerate(trunk):
            if isinstance(mod, nn.Linear):
                indices.append(i)
                covered.add(id(mod.weight))
                if mod.bias is not None:
                    covered.add(id(mod.bias))
        if covered != {id(p) for p in trunk.parameters()}:
            return None, None, None
        return trunk, head, indices

    def _ucb_widths_rowwise(self, obs: torch.Tensor) -> torch.Tensor:
        """``sqrt(g_a(x)^T Z^-1 g_a(x))`` by one backward pass per (row, action).

        Correct for any architecture and used as the fallback when the network
        has no linear head, but it costs ``n_rows * n_actions`` backward passes.
        :meth:`_ucb_widths` is the vectorised path and is tested against this.
        """
        widths = torch.empty(
            (len(obs), self.n_actions), dtype=torch.float32, device=self.device
        )
        for i in range(len(obs)):
            for a in range(self.n_actions):
                self.network.zero_grad()
                self.network(obs[i : i + 1])[0, a].backward()
                grad = torch.cat(
                    [p.grad.flatten() for p in self.network.parameters() if p.grad is not None]
                ).double()
                widths[i, a] = torch.sqrt((grad**2 / self.Z_diag).sum()).float()
        self.network.zero_grad()
        return widths

    def _ucb_widths(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-context confidence widths for every action, vectorised exactly.

        Equal to :meth:`_ucb_widths_rowwise` to floating-point tolerance, but
        it never forms a per-row gradient over the full parameter vector, which
        would be ``n_rows x 1.1M`` floats for KuaiRec.

        With ``f_a(x) = w_a . h(x) + c_a`` for the linear head ``(w, c)`` and
        trunk features ``h``, the gradient splits into a trunk part and a head
        part, and the head part is sparse -- only row ``a`` of the head has a
        non-zero gradient::

            g_a(x)^T Z^-1 g_a(x) = w_a^T M(x) w_a
                                   + sum_j h_j(x)^2 / Z[w_a_j]
                                   + 1 / Z[c_a]

        where ``M(x) = J(x) Z_trunk^-1 J(x)^T`` and ``J = d h / d theta_trunk``.
        ``M`` is ``hidden x hidden`` and does not depend on the action, so the
        trunk work is paid once per row rather than once per (row, action).

        ``M`` is assembled without ever materialising ``J``. For a linear layer
        with input ``a`` and pre-activation ``s``, ``d h_j / d W[m,k]`` is
        ``G[j,m] a_k`` with ``G = d h / d s``, so that layer contributes
        ``G diag(v) G^T`` with ``v_m = sum_k a_k^2 / Z[W_mk] + 1 / Z[b_m]``.
        ``G`` is only ``hidden x layer_width``; the dense ``J`` it replaces is
        ``hidden x n_layer_params``.
        """
        trunk, head, linear_indices = self._linear_trunk_and_head()
        if head is None:
            return self._ucb_widths_rowwise(obs)

        n_rows = len(obs)
        if n_rows == 0:
            return torch.empty((0, self.n_actions), dtype=torch.float32, device=self.device)

        z_of = {id(p): self.Z_diag[start:stop] for p, start, stop in self._parameter_slices()}
        hidden = head.in_features
        head_w = head.weight.detach()                                    # (A, H)
        inv_z_w = (1.0 / z_of[id(head.weight)]).view(self.n_actions, hidden).float()
        inv_z_b = (
            (1.0 / z_of[id(head.bias)]).float() if head.bias is not None else None
        )

        # Per-layer reciprocal-Z factors, precomputed once for all row blocks.
        layer_terms = []
        for i in linear_indices:
            mod = trunk[i]
            inv_w = (1.0 / z_of[id(mod.weight)]).view(mod.out_features, mod.in_features).float()
            inv_b = (
                (1.0 / z_of[id(mod.bias)]).float() if mod.bias is not None else None
            )
            layer_terms.append((i, mod, inv_w, inv_b))

        widest = max(mod.out_features for _, mod, _, _ in layer_terms)
        budget = max(int(self.ucb_memory_budget), 1 << 20)
        row_block = max(1, min(n_rows, budget // 2 // (4 * 3 * hidden * (widest + hidden))))
        act_block = max(1, min(self.n_actions, budget // 2 // (4 * hidden * hidden)))

        widths = torch.empty((n_rows, self.n_actions), dtype=torch.float32, device=self.device)
        self.network.eval()
        with torch.no_grad():
            for start in range(0, n_rows, row_block):
                block = obs[start : start + row_block]
                rows = len(block)
                m = torch.zeros((rows, hidden, hidden), dtype=torch.float32, device=self.device)
                activation = block
                cursor = 0
                for i, mod in enumerate(trunk):
                    if cursor < len(layer_terms) and layer_terms[cursor][0] == i:
                        _, _, inv_w, inv_b = layer_terms[cursor]
                        cursor += 1
                        pre = mod(activation)
                        after = trunk[i + 1 :]
                        jac = torch.func.vmap(
                            torch.func.jacrev(
                                lambda s, after=after: after(s.unsqueeze(0)).squeeze(0)
                            )
                        )(pre)                                            # (rows, H, out)
                        v = (activation * activation) @ inv_w.T            # (rows, out)
                        if inv_b is not None:
                            v = v + inv_b
                        m.baddbmm_(jac * v.unsqueeze(1), jac.transpose(1, 2))
                        activation = pre
                    else:
                        activation = mod(activation)
                features = activation                                      # h(x), (rows, H)

                m_flat = m.reshape(rows, hidden * hidden)
                sq_features = features * features
                for a0 in range(0, self.n_actions, act_block):
                    w = head_w[a0 : a0 + act_block]                        # (Ab, H)
                    # q[r, a] = w_a^T M_r w_a, as one large matmul against the
                    # outer products vec(w_a w_a^T); the (rows, H, Ab)
                    # intermediate of a bmm is avoided.
                    outer = (w.unsqueeze(2) * w.unsqueeze(1)).reshape(len(w), hidden * hidden)
                    q = m_flat @ outer.T
                    q += sq_features @ inv_z_w[a0 : a0 + act_block].T
                    if inv_z_b is not None:
                        q += inv_z_b[a0 : a0 + act_block]
                    widths[start : start + rows, a0 : a0 + act_block] = q.clamp_min_(0).sqrt_()
        return widths

    def _ucb_scores(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean prediction plus the per-context gradient confidence width."""
        self.network.eval()
        with torch.no_grad():
            mean = self.network(obs)
        return mean + self.nu * self._ucb_widths(obs)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        out = []
        for start in range(0, len(obs), 4096):
            out.append(self._ucb_scores(obs[start : start + 4096]).argmax(dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.network.state_dict(), "Z_diag": self.Z_diag,
             "n_actions": self.n_actions, "context_dim": self.context_dim, "nu": self.nu},
            path,
        )

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(state["state_dict"])
        self.Z_diag = state["Z_diag"].to(self.device)
        self.nu = state["nu"]


class DiscreteIQL(BasePolicy):
    """Implicit Q-Learning for discrete actions (Kostrikov et al., 2022).

    ``d3rlpy`` provides IQL for continuous actions only, so this is a direct
    implementation of the paper's three components:

    1. **Value network** trained by expectile regression towards ``Q(s, a)``:
       ``L_V = E[|tau - 1{u < 0}| u^2]`` where ``u = Q(s,a) - V(s)``. The
       asymmetric weight ``tau > 0.5`` makes ``V`` approximate an upper
       expectile of the Q distribution, which is what lets IQL estimate the
       value of the best in-distribution action without querying out-of-
       distribution actions.
    2. **Q network** trained on the TD target ``r + gamma * V(s')``.
    3. **Policy** extracted by advantage-weighted regression with weights
       ``exp(beta * (Q - V))``, clipped for numerical stability.

    The critical property is that no maximisation over unseen actions ever
    occurs, which is exactly what makes IQL safe for offline data.
    """

    name = "IQL"

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        hidden: int = 256,
        num_layers: int = 2,
        expectile: float = 0.7,
        beta: float = 3.0,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        epochs: int = 100,
        max_advantage_weight: float = 100.0,
        device: str = "cpu",
    ):
        if not 0 < expectile < 1:
            raise ValueError(f"expectile must be in (0, 1), got {expectile}")
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.device = torch.device(device)
        self.expectile = expectile
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_advantage_weight = max_advantage_weight

        self.q_network = _MLP(context_dim, hidden, n_actions, num_layers).to(self.device)
        self.value_network = _MLP(context_dim, hidden, 1, num_layers).to(self.device)
        self.policy_network = _MLP(context_dim, hidden, n_actions, num_layers).to(self.device)

    @staticmethod
    def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(diff > 0, expectile, 1.0 - expectile)
        return (weight * diff.pow(2)).mean()

    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray | None = None,
        terminals: np.ndarray | None = None,
    ) -> None:
        """Fit all three networks jointly.

        ``next_observations`` may be omitted for a single-step (contextual
        bandit) formulation, in which case ``gamma`` is effectively zero and the
        TD target reduces to the immediate reward. That choice is logged so it
        cannot silently change the semantics of the reported results.
        """
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rew = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)

        if next_observations is None:
            logger.info(
                "IQL fitting without next-state transitions: treating the problem as "
                "single-step (gamma effectively 0). This matches a contextual bandit "
                "formulation of the logged data."
            )
            next_obs = None
            done = torch.ones_like(rew)
        else:
            next_obs = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
            done = (
                torch.as_tensor(terminals, dtype=torch.float32, device=self.device)
                if terminals is not None
                else torch.zeros_like(rew)
            )

        q_opt = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        v_opt = torch.optim.Adam(self.value_network.parameters(), lr=self.learning_rate)
        pi_opt = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        n = len(obs)
        for epoch in range(self.epochs):
            permutation = torch.randperm(n, device=self.device)
            totals = np.zeros(3)
            n_batches = 0
            for start in range(0, n, self.batch_size):
                idx = permutation[start : start + self.batch_size]
                b_obs, b_act, b_rew = obs[idx], act[idx], rew[idx]

                # 1. Value network via expectile regression on the frozen Q.
                with torch.no_grad():
                    q_sa = self.q_network(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
                value = self.value_network(b_obs).squeeze(1)
                v_loss = self._expectile_loss(q_sa - value, self.expectile)
                v_opt.zero_grad()
                v_loss.backward()
                v_opt.step()

                # 2. Q network via TD on the frozen V.
                with torch.no_grad():
                    if next_obs is None:
                        target = b_rew
                    else:
                        next_value = self.value_network(next_obs[idx]).squeeze(1)
                        target = b_rew + self.gamma * (1.0 - done[idx]) * next_value
                q_pred = self.q_network(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
                q_loss = F.mse_loss(q_pred, target)
                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()

                # 3. Policy via advantage-weighted regression.
                with torch.no_grad():
                    advantage = (
                        self.q_network(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
                        - self.value_network(b_obs).squeeze(1)
                    )
                    weights = torch.clamp(
                        torch.exp(self.beta * advantage), max=self.max_advantage_weight
                    )
                log_probs = F.log_softmax(self.policy_network(b_obs), dim=1)
                pi_loss = -(weights * log_probs.gather(1, b_act.unsqueeze(1)).squeeze(1)).mean()
                pi_opt.zero_grad()
                pi_loss.backward()
                pi_opt.step()

                totals += [v_loss.item(), q_loss.item(), pi_loss.item()]
                n_batches += 1

            if (epoch + 1) % max(1, self.epochs // 10) == 0:
                v, q, p = totals / max(1, n_batches)
                logger.info(
                    "IQL epoch %d/%d | V=%.5f Q=%.5f pi=%.5f", epoch + 1, self.epochs, v, q, p
                )

    #: Cells (rows x actions) per forward pass in :meth:`predict`. The policy
    #: network emits one logit per action, so the ACTIVATION is
    #: ``n_rows x n_actions`` regardless of how small the network is: at
    #: KuaiRec's 4,676,570 evaluation rounds over 3,327 actions that is 58 GiB
    #: in one tensor, which is what an unchunked forward pass tried to allocate.
    #: 2e7 cells is ~76 MB in float32. Chunking is numerically inert here --
    #: ``argmax`` is per row and independent of every other row.
    PREDICT_CELL_BUDGET = 20_000_000

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Greedy action per row, evaluated in bounded forward passes.

        LinUCB, NeuralUCB and the d3rlpy agents already chunk their prediction;
        this one did not, and was the last place a full-length forward pass
        could be issued.
        """
        self.policy_network.eval()
        n_rows = len(observations)
        if n_rows == 0:
            return np.empty(0, dtype=np.int64)

        rows_per_chunk = max(1, self.PREDICT_CELL_BUDGET // max(1, self.n_actions))
        out = np.empty(n_rows, dtype=np.int64)
        with torch.no_grad():
            for start in range(0, n_rows, rows_per_chunk):
                stop = min(start + rows_per_chunk, n_rows)
                block = torch.as_tensor(
                    observations[start:stop], dtype=torch.float32, device=self.device
                )
                out[start:stop] = (
                    self.policy_network(block).argmax(dim=1).cpu().numpy()
                )
        return out

    #: Above this many (rows x actions) cells, the dense softmax is not materialised
    #: and the greedy policy is evaluated instead. 5e7 cells is ~400 MB in float64.
    #: This is Phase 4's own budget, imported rather than repeated: if the two
    #: numbers could drift apart, IQL could hand Phase 4 a distribution that
    #: Phase 4's guard then refuses, and the run would die on a config that used
    #: to work.
    DENSE_CELL_BUDGET = DENSE_ACTION_DIST_CELL_BUDGET

    def distribution_is_dense(self, n_rounds: int) -> bool:
        """True exactly when :meth:`action_distribution` returns the dense softmax."""
        return n_rounds * self.n_actions <= self.DENSE_CELL_BUDGET

    def action_distribution(self, observations: np.ndarray):
        """IQL yields a stochastic policy; return its softmax where that is feasible.

        For a large catalogue the dense softmax exceeds
        :attr:`DENSE_CELL_BUDGET`. Rather than silently truncating it, the greedy
        policy is evaluated instead and the substitution is logged, because the
        two are different policies with different values.
        """
        from ..evaluation.ope import DeterministicPolicy

        n_cells = len(observations) * self.n_actions
        if n_cells > self.DENSE_CELL_BUDGET:
            logger.warning(
                "IQL: the dense softmax over %d rounds x %d actions would need %.1f GB. "
                "Evaluating the greedy (argmax) policy instead. This is a different policy "
                "from the stochastic one and must be described as such.",
                len(observations), self.n_actions, n_cells * 8 / 1e9,
            )
            return DeterministicPolicy(self.predict(observations), self.n_actions)

        self.policy_network.eval()
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return F.softmax(self.policy_network(obs), dim=1).cpu().numpy().astype(np.float64)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"q": self.q_network.state_dict(), "v": self.value_network.state_dict(),
             "pi": self.policy_network.state_dict(), "n_actions": self.n_actions,
             "context_dim": self.context_dim, "expectile": self.expectile}, path,
        )

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(state["q"])
        self.value_network.load_state_dict(state["v"])
        self.policy_network.load_state_dict(state["pi"])


class D3RLPyPolicy(BasePolicy):
    """Adapter around ``d3rlpy``'s discrete offline RL algorithms.

    Covers DQN (unconstrained baseline), CQL (conservative value penalty) and
    BCQ (batch-constrained action selection). These are deterministic greedy
    policies, so :meth:`action_distribution` returns one-hot rows -- which is
    their true distribution, not an approximation.

    Note on naming: ``d3rlpy`` exposes DQN as ``DQNConfig``, not
    ``DiscreteDQNConfig``. DQN is inherently discrete-action, so there is no
    separate discrete variant, and referencing ``DiscreteDQNConfig`` raises
    ``AttributeError``.
    """

    SUPPORTED = ("DQN", "DoubleDQN", "CQL", "BCQ")

    def __init__(
        self,
        algorithm: str,
        n_actions: int,
        context_dim: int,
        hidden_units: int = 256,
        num_layers: int = 2,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        epochs: int = 100,
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        cql_alpha: float = 5.0,
        bcq_action_flexibility: float = 0.3,
        device: str = "cpu",
        seed: int = 42,
    ):
        if algorithm not in self.SUPPORTED:
            raise ValueError(
                f"unsupported algorithm {algorithm!r}; supported: {self.SUPPORTED}. "
                f"IQL is implemented separately in DiscreteIQL because d3rlpy provides "
                f"IQL for continuous actions only."
            )
        self.name = algorithm
        self.algorithm = algorithm
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.epochs = epochs
        self.batch_size = batch_size
        # d3rlpy's torch_utility.map_location does ``_, index = device.split(":")``
        # when reloading a checkpoint, so a bare "cuda" raises ValueError there.
        # resolve_device already returns an indexed device, but this class is the
        # boundary with the library that imposes the requirement, so it enforces
        # it here too rather than trusting every future caller.
        self.device = "cuda:0" if device == "cuda" else device
        self.seed = seed
        self._algo: Any = None
        self._built = False

        self._config_kwargs = {
            "hidden_units": hidden_units, "num_layers": num_layers,
            "learning_rate": learning_rate, "batch_size": batch_size, "gamma": gamma,
            "target_update_interval": target_update_interval, "cql_alpha": cql_alpha,
            "bcq_action_flexibility": bcq_action_flexibility,
        }

    def _build_config(self):
        import d3rlpy
        from d3rlpy.models.encoders import VectorEncoderFactory

        kw = self._config_kwargs
        encoder = VectorEncoderFactory(hidden_units=[kw["hidden_units"]] * kw["num_layers"])
        common = {
            "batch_size": kw["batch_size"],
            "learning_rate": kw["learning_rate"],
            "gamma": kw["gamma"],
            "encoder_factory": encoder,
            "target_update_interval": kw["target_update_interval"],
        }

        if self.algorithm == "DQN":
            return d3rlpy.algos.DQNConfig(**common)
        if self.algorithm == "DoubleDQN":
            return d3rlpy.algos.DoubleDQNConfig(**common)
        if self.algorithm == "CQL":
            return d3rlpy.algos.DiscreteCQLConfig(**common, alpha=kw["cql_alpha"])
        return d3rlpy.algos.DiscreteBCQConfig(
            **common, action_flexibility=kw["bcq_action_flexibility"]
        )

    def _make_dataset(self, observations, actions, rewards, terminals=None):
        """Build an MDPDataset covering the full action space.

        ``d3rlpy`` infers the action space size from the actions present. If a
        chunk happens not to contain the highest action index the network is
        built with too few outputs, so the action space is asserted rather than
        patched with a synthetic transition appended to the real data.
        """
        from d3rlpy.dataset import MDPDataset

        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)

        observed = int(actions.max(initial=-1)) + 1
        if observed > self.n_actions:
            raise ValueError(
                f"action index {observed - 1} exceeds declared action space {self.n_actions}"
            )

        if terminals is None:
            # Each logged interaction is an independent single-step episode.
            # Marking every row terminal states that explicitly, instead of the
            # all-zeros array that leaves d3rlpy with one unbounded episode and
            # meaningless bootstrapped next-state values.
            terminals = np.ones(len(actions), dtype=np.float32)
        else:
            terminals = np.asarray(terminals, dtype=np.float32)

        return MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            action_space=__import__("d3rlpy").ActionSpace.DISCRETE,
            action_size=self.n_actions,
        )

    def fit(self, observations, actions, rewards, terminals=None) -> None:
        import d3rlpy

        d3rlpy.seed(self.seed)
        dataset = self._make_dataset(observations, actions, rewards, terminals)

        if self._algo is None:
            self._algo = self._build_config().create(device=self.device)
        algo = self._algo
        if not self._built:
            algo.build_with_dataset(dataset)
            self._built = True

        steps_per_epoch = max(1, len(actions) // self.batch_size)
        # NoopAdapter: this project writes its own structured logs and result
        # files, so d3rlpy's per-run directories would only litter the working
        # directory with a second, inconsistent record of the same run.
        algo.fit(
            dataset,
            n_steps=steps_per_epoch * self.epochs,
            n_steps_per_epoch=steps_per_epoch,
            show_progress=False,
            logger_adapter=d3rlpy.logging.NoopAdapterFactory(),
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        if self._algo is None:
            raise RuntimeError(f"{self.name} must be fitted or loaded before predict()")
        observations = np.asarray(observations, dtype=np.float32)
        out = [
            self._algo.predict(observations[start : start + 4096])
            for start in range(0, len(observations), 4096)
        ]
        return np.concatenate(out).astype(np.int64) if out else np.empty(0, dtype=np.int64)

    def save(self, path: Path) -> None:
        if self._algo is None:
            raise RuntimeError(f"{self.name} has not been fitted; nothing to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._algo.save_model(str(path))

    def load(self, path: Path) -> None:
        if self._algo is None:
            raise RuntimeError(
                "build the algorithm with fit() or build_for_loading() before load()"
            )
        self._algo.load_model(str(path))

    def build_for_loading(self, observations: np.ndarray, actions: np.ndarray) -> None:
        """Construct the network graph so saved weights can be loaded into it."""
        dataset = self._make_dataset(
            observations[:2], actions[:2], np.zeros(2, dtype=np.float32)
        )
        algo = self._build_config().create(device=self.device)
        algo.build_with_dataset(dataset)
        self._algo = algo
        self._built = True


def build_policy(name: str, n_actions: int, context_dim: int, config, device: str, seed: int) -> BasePolicy:
    """Instantiate an agent by name from the Phase 3 configuration.

    Raises:
        ValueError: for an unknown agent. There is no fallback that silently
            substitutes a different algorithm.
    """
    if name == "Random":
        return RandomPolicy(n_actions=n_actions, seed=seed)
    if name == "LinUCB":
        return LinUCB(n_actions, context_dim, alpha=config.linucb_alpha,
                      ridge_lambda=config.linucb_ridge_lambda)
    if name == "NeuralUCB":
        return NeuralUCB(n_actions, context_dim, hidden=config.neuralucb_hidden,
                         lambda_=config.neuralucb_lambda, nu=config.neuralucb_nu,
                         epochs=config.neuralucb_epochs, learning_rate=config.learning_rate,
                         batch_size=config.batch_size, device=device)
    if name == "IQL":
        return DiscreteIQL(n_actions, context_dim, hidden=config.hidden_units,
                           num_layers=config.num_layers, expectile=config.iql_expectile,
                           beta=config.iql_beta, gamma=config.gamma,
                           learning_rate=config.learning_rate, batch_size=config.batch_size,
                           epochs=config.epochs, device=device)
    if name in D3RLPyPolicy.SUPPORTED:
        return D3RLPyPolicy(name, n_actions, context_dim, hidden_units=config.hidden_units,
                            num_layers=config.num_layers, learning_rate=config.learning_rate,
                            batch_size=config.batch_size, epochs=config.epochs,
                            gamma=config.gamma, target_update_interval=config.target_update_interval,
                            cql_alpha=config.cql_alpha,
                            bcq_action_flexibility=config.bcq_action_flexibility,
                            device=device, seed=seed)

    raise ValueError(
        f"unknown agent {name!r}. Available: Random, LinUCB, NeuralUCB, IQL, "
        f"{', '.join(D3RLPyPolicy.SUPPORTED)}"
    )
