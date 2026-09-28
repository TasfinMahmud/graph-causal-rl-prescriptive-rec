"""Phase 1: train graph encoders and export node embeddings.

Two properties matter here and neither held previously.

**The output is embeddings.** The previous implementation saved
``model.state_dict()`` into a file named after the embedding directory, so no
downstream phase could ever have loaded a graph representation -- there were
none on disk. This phase runs the encoder and saves the resulting node
embedding matrix, keyed so Phase 3 can look up a user's row.

**Architectures are compared on held-out data.** Validation BPR loss on a set of
edges withheld from training decides early stopping and the reported ranking.
Ranking by training loss measures fit to the training edges rather than
representation quality, and is not comparable across models trained with
different batch configurations.

**Two propagation graphs, used deliberately.** Splitting the supervision edges is
not enough on its own: if message passing runs over the *full* edge set, a
validation edge is still present in the graph that produces the embeddings used
to score it, and BPR "validation" loss degenerates into reading back an edge the
encoder can see. So:

* ``train_message_edges`` -- the training edges only, made undirected and with
  any held-out pair removed in *both* directions. Used for every training step
  and for every validation evaluation. This is what makes the reported
  validation BPR loss, and therefore the early-stopping point and the Phase 1
  ranking, an honest held-out measurement.
* ``graph.edge_index`` -- the full edge set. Used once, by
  :func:`export_embeddings`, for the artefact Phase 2 and Phase 3 consume.

The asymmetry is intentional and is the standard transductive arrangement: the
representation used at *inference* may use every edge available at training
time, and the Phase 1 validation edges are a model-selection device carved out
of the dataset's own training split, not held-out evaluation data. The Phase 4
test split never enters either graph. What must not happen -- and no longer
does -- is a validation edge contributing to the representation that scores it.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import ExperimentConfig, resolve_device
from ..data.graphs import InteractionGraph, split_edges, to_undirected
from ..logging_utils import get_logger
from ..models.gnn import bpr_loss, build_gnn

logger = get_logger(__name__)


@dataclass(frozen=True)
class NodeEmbeddings:
    """A Phase 1 artefact: the node embedding matrix plus what is needed to index it."""

    embeddings: torch.Tensor
    num_users: int
    num_items: int
    architecture: str
    dataset: str
    embedding_dim: int
    num_layers: int
    seed: int

    def for_users(self, user_index: np.ndarray) -> np.ndarray:
        """Return the embedding row for each entry of ``user_index``.

        Raises:
            IndexError: if an index addresses a row outside the user block.
                Clipping instead would silently map several users onto one state.
        """
        index = np.asarray(user_index, dtype=np.int64)
        if index.size and (index.max() >= self.num_users or index.min() < 0):
            raise IndexError(
                f"user index range [{index.min()}, {index.max()}] falls outside the "
                f"{self.num_users} user rows of this embedding table"
            )
        return self.embeddings.numpy()[index].astype(np.float32)


@dataclass
class GNNTrainingResult:
    """Everything needed to report a Phase 1 row, including how it was produced."""

    architecture: str
    dataset: str
    best_val_bpr_loss: float
    final_train_bpr_loss: float
    best_epoch: int
    epochs_run: int
    early_stopped: bool
    training_seconds: float
    n_parameters: int
    embedding_path: str
    seed: int
    peak_memory_mb: float | None = None

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def _sample_negative_items(
    n_samples: int, num_users: int, num_items: int, generator: torch.Generator
) -> torch.Tensor:
    """Draw uniform random item nodes as BPR negatives."""
    return torch.randint(
        num_users, num_users + num_items, (n_samples,), generator=generator
    )


def build_message_graph(
    train_edges: torch.Tensor, val_edges: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """The propagation graph for training and validation: training edges only.

    ``to_undirected`` restores the reverse direction that Phase 1 strips when it
    takes the forward (user -> item) edges as supervision. Any edge whose
    *undirected* pair is held out is then removed, because the reverse of a
    validation edge carries it into the propagation graph exactly as the edge
    itself does -- and ``to_undirected`` on the whole graph put it there.

    On a purely bipartite graph the second step removes nothing, since a forward
    edge and its reverse are never split apart. It matters for the user-user
    social edges of KuaiRec, which appear in both directions in
    ``graph.edge_index`` and so can land on opposite sides of the split; the
    conservative choice is to hold out the undirected edge.
    """
    edges = to_undirected(train_edges)
    if val_edges.numel() == 0 or edges.numel() == 0:
        return edges
    held_out = to_undirected(val_edges)
    # Encode each directed edge as a single integer so membership is one isin().
    def keys(e: torch.Tensor) -> torch.Tensor:
        return e[0] * num_nodes + e[1]

    return edges[:, ~torch.isin(keys(edges), keys(held_out))]


def _evaluate(
    model: torch.nn.Module,
    features: torch.Tensor,
    message_edges: torch.Tensor,
    supervision_edges: torch.Tensor,
    graph: InteractionGraph,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    """Mean BPR loss on ``supervision_edges``, propagating over ``message_edges``.

    The two edge sets must be disjoint, in both directions. Callers pass the
    training-only propagation graph built by :func:`build_message_graph`, so a
    validation edge never contributes to the representation used to score it.
    Passing the full edge set here instead makes the returned number a
    link-prediction score for an edge the encoder can already see: trivially
    optimistic, and optimistic by an architecture-dependent amount, since models
    differ in how strongly a present edge propagates into its own endpoints.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(features, message_edges)
        users = supervision_edges[0]
        positives = supervision_edges[1]
        negatives = _sample_negative_items(
            users.size(0), graph.num_users, graph.num_items, generator
        ).to(device)
        return float(bpr_loss(embeddings, users, positives, negatives).item())


def _artefact_paths(config: ExperimentConfig, architecture: str, seed: int) -> tuple[Path, Path]:
    """Where this encoder keeps its embeddings and its metrics sidecar.

    Keyed by a FINGERPRINT of everything that produced it -- the data, the
    architecture, and every GNN hyperparameter -- not by ``dataset.name`` alone.

    Keying on ``(dataset.name, architecture, seed)`` would not be enough:
    ``obd_fast.yaml`` inherits ``dataset.name: obd``, so a 300,000-row /
    20-epoch run and a 2,000,000-row / 50-epoch run would resolve to the same
    path, and whichever ran second would consume the first one's embeddings
    while reporting its own settings. Deliberately NOT keyed by
    ``experiment_name``: the three
    ablation arms differ only in ``rl.*``, which Phase 1 does not read, so they
    should and do share one encoder.
    """
    from ..cli import embedding_fingerprint

    out = Path(config.paths.embeddings)
    stem = f"{config.dataset.name}_{architecture}_seed{seed}"
    stem = f"{stem}_{embedding_fingerprint(config, architecture, seed)}"
    return out / f"{stem}_embeddings.pt", out / f"{stem}_metrics.json"


def embedding_path(config: ExperimentConfig, architecture: str, seed: int) -> Path:
    """The one place any phase may learn where an encoder's embeddings live.

    Every phase resolves the path through this function rather than rebuilding
    it from a format string of its own. Independent constructions of the same
    filename are how a writer and a reader drift apart without anything failing
    loudly.
    """
    return _artefact_paths(config, architecture, seed)[0]


def _reuse_existing(
    config: ExperimentConfig, architecture: str, seed: int
) -> GNNTrainingResult | None:
    """Return a previous run's result if a complete, loadable artefact exists.

    Three things must hold, and all three are checked. Both files must be
    present; the metrics must parse into a ``GNNTrainingResult``; and the
    ``.pt`` must actually load and carry embeddings of the width this config
    asks for. Checking only that the two files exist is not sufficient: a
    zero-byte ``.pt`` would be reported as a successful reuse, with the failure
    surfacing hours later in a different phase.
    """
    emb, metrics = _artefact_paths(config, architecture, seed)
    if not (emb.exists() and metrics.exists()):
        return None
    try:
        result = GNNTrainingResult(**json.loads(metrics.read_text(encoding="utf-8")))
        payload = torch.load(emb, map_location="cpu", weights_only=False)
        stored = payload["embeddings"]
        if stored.size(1) != config.gnn.embedding_dim:
            raise ValueError(
                f"embedding width {stored.size(1)} != configured "
                f"gnn.embedding_dim {config.gnn.embedding_dim}"
            )
        if not torch.isfinite(stored).all():
            raise ValueError("stored embeddings contain non-finite values")
    except Exception as exc:
        logger.warning(
            "cannot reuse %s (%s); retraining %s", emb.name, exc, architecture
        )
        return None
    logger.info(
        "[%s] %s reusing embeddings from an earlier run with identical settings "
        "(seed %d) -- not retraining", config.dataset.name, architecture, seed,
    )
    return result


def train_single_gnn(
    architecture: str,
    graph: InteractionGraph,
    config: ExperimentConfig,
    seed: int,
    device: str,
    output_dir: Path,
) -> GNNTrainingResult:
    """Train one architecture with early stopping on validation BPR loss.

    Raises:
        RuntimeError: on out-of-memory. OOM is not caught and skipped: doing so
            silently trains a model on a subset of its batches while reporting a
            loss that is not comparable to models that saw every batch.
    """
    from ..cli import force_rebuild

    if not force_rebuild():
        cached = _reuse_existing(config, architecture, seed)
        if cached is not None:
            return cached

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device_obj = torch.device(device)

    # Only user->item edges carry supervision; the reverse direction exists for
    # message passing but would duplicate every training pair.
    forward_mask = graph.edge_index[0] < graph.num_users
    supervision_edges = graph.edge_index[:, forward_mask]
    train_edges, val_edges = split_edges(supervision_edges, val_fraction=0.1, seed=seed)

    if graph.node_features is None:
        raise ValueError(
            "the graph has no node features. build_bipartite_graph seeds a random "
            "initialisation when none are supplied, so this indicates the graph was "
            "constructed by other means."
        )

    # Propagation graph for training AND for validation scoring: training edges
    # only. See this module's docstring for why the export graph below differs.
    train_message_edges = build_message_graph(
        train_edges, val_edges, graph.num_nodes
    ).to(device_obj)
    full_message_edges = graph.edge_index.to(device_obj)
    train_edges = train_edges.to(device_obj)
    val_edges = val_edges.to(device_obj)
    features = graph.node_features.to(device_obj)

    model = build_gnn(architecture, features.size(1), config.gnn).to(device_obj)
    n_parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.gnn.learning_rate, weight_decay=config.gnn.weight_decay
    )

    logger.info(
        "[%s] %s: %d parameters, %d train edges, %d val edges, %d layers, dim %d",
        config.dataset.name, architecture, n_parameters, train_edges.size(1),
        val_edges.size(1), config.gnn.num_layers, config.gnn.embedding_dim,
    )
    logger.info(
        "[%s] %s: propagating over %d train-only message edges during training and "
        "validation (%d in the full graph, used only for the exported embeddings)",
        config.dataset.name, architecture, train_message_edges.size(1),
        full_message_edges.size(1),
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    final_train_loss = float("nan")
    started = time.time()

    n_train = train_edges.size(1)
    batch_size = min(config.gnn.node_batch_size, n_train)

    for epoch in range(config.gnn.epochs):
        model.train()
        permutation = torch.randperm(n_train, generator=generator).to(device_obj)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, n_train, batch_size):
            index = permutation[start : start + batch_size]
            users = train_edges[0, index]
            positives = train_edges[1, index]
            negatives = _sample_negative_items(
                users.size(0), graph.num_users, graph.num_items, generator
            ).to(device_obj)

            optimizer.zero_grad()
            embeddings = model(features, train_message_edges)
            loss = bpr_loss(embeddings, users, positives, negatives)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        final_train_loss = epoch_loss / max(1, n_batches)
        val_loss = _evaluate(
            model, features, train_message_edges, val_edges, graph, generator, device_obj
        )

        improved = val_loss < best_val - config.gnn.early_stopping_min_delta
        if improved:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.info(
            "[%s] %s epoch %d/%d train=%.6f val=%.6f%s",
            config.dataset.name, architecture, epoch + 1, config.gnn.epochs,
            final_train_loss, val_loss, "  <- best" if improved else "",
        )

        if epochs_without_improvement >= config.gnn.early_stopping_patience:
            logger.info(
                "[%s] %s early stopping at epoch %d (no improvement for %d epochs)",
                config.dataset.name, architecture, epoch + 1, config.gnn.early_stopping_patience,
            )
            break

    epochs_run = epoch + 1
    early_stopped = epochs_without_improvement >= config.gnn.early_stopping_patience

    if best_state is not None:
        model.load_state_dict(best_state)

    embedding_path = export_embeddings(
        model, features, full_message_edges, graph, architecture, config, output_dir, seed
    )

    peak_memory = None
    if device_obj.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device_obj) / (1024**2)
        torch.cuda.reset_peak_memory_stats(device_obj)

    result = GNNTrainingResult(
        architecture=architecture,
        dataset=config.dataset.name,
        best_val_bpr_loss=best_val,
        final_train_bpr_loss=final_train_loss,
        best_epoch=best_epoch + 1,
        epochs_run=epochs_run,
        early_stopped=early_stopped,
        training_seconds=time.time() - started,
        n_parameters=n_parameters,
        embedding_path=str(embedding_path),
        seed=seed,
        peak_memory_mb=peak_memory,
    )
    _, metrics_path = _artefact_paths(config, architecture, seed)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = metrics_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(dataclasses.asdict(result), indent=2, default=str), encoding="utf-8")
    tmp.replace(metrics_path)
    return result


def export_embeddings(
    model: torch.nn.Module,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    graph: InteractionGraph,
    architecture: str,
    config: ExperimentConfig,
    output_dir: Path,
    seed: int,
) -> Path:
    """Run the encoder and persist the node embedding matrix.

    Saves the embeddings themselves -- not the model weights -- together with
    the user/item counts needed to slice them, so Phase 3 can index a user's
    representation without re-running the encoder.

    ``edge_index`` here is the FULL graph, unlike the train-only graph used for
    training and validation scoring. Nothing in this artefact is a score for one
    of the held-out edges: Phase 2 and Phase 3 read a user's row as a state
    vector, they do not predict the Phase 1 validation edges. Every edge in the
    full graph comes from the dataset's training split, so this leaks nothing
    into Phase 4. Restricting the export to the training edges would throw away
    10% of the observed structure from the representation for no gain.
    """
    model.eval()
    with torch.no_grad():
        embeddings = model(features, edge_index).cpu()

    if not torch.isfinite(embeddings).all():
        raise RuntimeError(
            f"{architecture} produced non-finite embeddings; refusing to export them"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # output_dir is honoured for compatibility, but the FILENAME comes from the
    # shared helper so the writer cannot diverge from the readers.
    path = output_dir / embedding_path(config, architecture, seed).name

    torch.save(
        {
            "embeddings": embeddings,
            "num_users": graph.num_users,
            "num_items": graph.num_items,
            "architecture": architecture,
            "dataset": config.dataset.name,
            "embedding_dim": embeddings.size(1),
            "num_layers": config.gnn.num_layers,
            "seed": seed,
        },
        path,
    )
    logger.info("Exported embeddings %s -> %s", tuple(embeddings.shape), path.name)
    return path


def load_embeddings(path: Path) -> NodeEmbeddings:
    """Load an embedding artefact written by :func:`export_embeddings`.

    Raises:
        FileNotFoundError: with guidance, rather than letting Phase 3 fall back
            to raw features and silently change what the reported results mean.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"embeddings not found at {path}. Run Phase 1 before Phase 3, or set "
            f"rl.state_source='raw_features' if a tabular state is intended."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise ValueError(
            f"{path} does not contain an 'embeddings' key. A file holding model "
            f"weights rather than embeddings cannot be used as a state."
        )
    return NodeEmbeddings(
        embeddings=payload["embeddings"],
        num_users=int(payload["num_users"]),
        num_items=int(payload["num_items"]),
        architecture=str(payload["architecture"]),
        dataset=str(payload["dataset"]),
        embedding_dim=int(payload["embedding_dim"]),
        num_layers=int(payload["num_layers"]),
        seed=int(payload["seed"]),
    )


def run_phase1(
    graph: InteractionGraph, config: ExperimentConfig, seed: int | None = None
) -> list[GNNTrainingResult]:
    """Train every configured architecture and write the comparison table."""
    seed = seed if seed is not None else config.seed
    device = resolve_device(config.device)
    logger.info("Phase 1 on device %s | architectures: %s", device, config.gnn.architectures)

    results: list[GNNTrainingResult] = []
    for architecture in config.gnn.architectures:
        result = train_single_gnn(
            architecture, graph, config, seed, device, Path(config.paths.embeddings)
        )
        results.append(result)
        logger.info(
            "[%s] %s done: best val BPR %.6f at epoch %d (%.1fs)",
            config.dataset.name, architecture, result.best_val_bpr_loss,
            result.best_epoch, result.training_seconds,
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    output = Path(config.paths.results) / f"phase1_gnn_{config.dataset.name}_seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump([r.as_row() for r in results], handle, indent=2)
    logger.info("Phase 1 results -> %s", output)

    ranked = sorted(results, key=lambda r: r.best_val_bpr_loss)
    logger.info(
        "Phase 1 ranking by validation BPR loss: %s",
        ", ".join(f"{r.architecture}={r.best_val_bpr_loss:.6f}" for r in ranked),
    )
    return results
