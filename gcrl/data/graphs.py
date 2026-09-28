"""Bipartite interaction graph construction.

Nodes are users followed by items in one index space: user ``u`` is node ``u``
and item ``i`` is node ``n_users + i``. Edges are made undirected so messages
flow in both directions, which a bipartite recommender graph requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class InteractionGraph:
    """A bipartite user-item graph with optional extra user-user edges."""

    edge_index: torch.Tensor
    num_users: int
    num_items: int
    node_features: torch.Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return self.num_users + self.num_items

    def user_slice(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings[: self.num_users]

    def item_slice(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings[self.num_users :]

    def describe(self) -> str:
        return (
            f"graph: {self.num_nodes:,} nodes ({self.num_users:,} users + "
            f"{self.num_items:,} items), {self.edge_index.size(1):,} directed edges, "
            f"features={'yes' if self.node_features is not None else 'random init'}"
        )


def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """Append reversed edges and drop duplicates."""
    reversed_edges = edge_index.flip(0)
    combined = torch.cat([edge_index, reversed_edges], dim=1)
    return torch.unique(combined, dim=1)


def build_bipartite_graph(
    user_index: np.ndarray,
    item_index: np.ndarray,
    num_users: int,
    num_items: int,
    node_features: np.ndarray | None = None,
    social_edges: np.ndarray | None = None,
    user_id_to_index: dict | None = None,
    embedding_dim: int = 64,
    seed: int = 42,
) -> InteractionGraph:
    """Assemble the interaction graph.

    Args:
        social_edges: Optional ``(2, n)`` user-user edges in *raw* id space,
            mapped through ``user_id_to_index``. Edges referencing users outside
            the interaction log are dropped and counted.
        node_features: Optional ``(num_nodes, d)`` features. When absent, nodes
            get a seeded random initialisation -- seeded because the previous
            implementation drew fresh random features on every run, which alone
            made results irreproducible regardless of any other seeding.

    Raises:
        ValueError: if indices fall outside the declared node counts.
    """
    user_index = np.asarray(user_index, dtype=np.int64)
    item_index = np.asarray(item_index, dtype=np.int64)

    if user_index.max(initial=-1) >= num_users:
        raise ValueError(f"user index {user_index.max()} >= num_users {num_users}")
    if item_index.max(initial=-1) >= num_items:
        raise ValueError(f"item index {item_index.max()} >= num_items {num_items}")

    edges = torch.stack([
        torch.as_tensor(user_index, dtype=torch.long),
        torch.as_tensor(item_index + num_users, dtype=torch.long),
    ])

    if social_edges is not None and user_id_to_index is not None:
        mapped_src, mapped_dst, dropped = [], [], 0
        for source, target in zip(social_edges[0], social_edges[1], strict=True):
            s, t = user_id_to_index.get(int(source)), user_id_to_index.get(int(target))
            if s is None or t is None:
                dropped += 1
                continue
            mapped_src.append(s)
            mapped_dst.append(t)
        if mapped_src:
            social = torch.stack([
                torch.as_tensor(mapped_src, dtype=torch.long),
                torch.as_tensor(mapped_dst, dtype=torch.long),
            ])
            edges = torch.cat([edges, social], dim=1)
            logger.info(
                "Added %d social edges (%d dropped: endpoint not in the interaction log)",
                len(mapped_src), dropped,
            )
        elif dropped:
            logger.warning("All %d social edges dropped; no endpoints matched the log", dropped)

    edge_index = to_undirected(edges)
    num_nodes = num_users + num_items

    if node_features is not None:
        features = torch.as_tensor(node_features, dtype=torch.float32)
        if features.size(0) != num_nodes:
            raise ValueError(
                f"node_features has {features.size(0)} rows but the graph has {num_nodes} nodes"
            )
    else:
        generator = torch.Generator().manual_seed(seed)
        features = torch.randn(num_nodes, embedding_dim, generator=generator)
        logger.info(
            "No node features supplied; initialised %d x %d random features with seed %d",
            num_nodes, embedding_dim, seed,
        )

    graph = InteractionGraph(edge_index, num_users, num_items, features)
    logger.info(graph.describe())
    return graph


def split_edges(
    edge_index: torch.Tensor, val_fraction: float = 0.1, seed: int = 42
) -> tuple:
    """Hold out a fraction of edges for validation.

    Phase 1 needs a validation signal for early stopping. Ranking architectures
    by their *training* loss -- as the previous implementation did -- measures
    fit to the training edges, not representation quality.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n_edges = edge_index.size(1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_edges, generator=generator)
    n_val = max(1, int(n_edges * val_fraction))
    return edge_index[:, permutation[n_val:]], edge_index[:, permutation[:n_val]]
