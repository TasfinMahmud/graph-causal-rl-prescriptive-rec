"""Graph neural network encoders for user-item interaction graphs.

Message passing is implemented directly with ``torch.index_add`` rather than
via ``torch_geometric``. PyG couples tightly to specific torch builds and its
compiled extensions are a recurring source of platform-specific failures -- the
previous implementation carried a hand-rolled sampler explicitly to work around
"Windows pyg-lib DLL crashes". Native scatter operations remove that dependency
entirely and are exactly as correct.

Every architecture below implements its published propagation rule. Where a
model is a simplification of its paper, the docstring says so explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 (conventional torch alias)

from ..logging_utils import get_logger

logger = get_logger(__name__)


def compute_symmetric_norm(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Symmetric normalisation coefficient ``1 / sqrt(deg_src * deg_dst)`` per edge.

    Isolated nodes would divide by zero, so degrees are clamped at 1; such nodes
    receive no messages anyway and keep their self-representation.
    """
    row, col = edge_index[0], edge_index[1]
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    degree.index_add_(0, row, torch.ones_like(row, dtype=torch.float32))
    degree = degree.clamp(min=1.0)
    return (degree[row].pow(-0.5) * degree[col].pow(-0.5)).unsqueeze(-1)


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return ``edge_index`` with exactly one self-loop per node (``A~ = A + I``).

    Any self-loop already present is dropped first, so a node cannot end up with
    two and have its own term counted twice.

    This is deliberately *not* applied in ``gcrl.data.graphs.to_undirected``.
    Only GCN and GAT want self-loops: LightGCN's published rule is
    ``e_u^(k+1) = sum_{i in N(u)} 1/sqrt(|N_u||N_i|) e_i^(k)`` with no
    self-connection, NGCF carries its self term separately as ``W1 e_u``, and
    GraphSAGE has a dedicated ``W_self``. Adding self-loops to the shared graph
    would silently change all three.
    """
    keep = edge_index[0] != edge_index[1]
    loops = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype)
    return torch.cat([edge_index[:, keep], loops.repeat(2, 1)], dim=1)


def aggregate(
    messages: torch.Tensor, index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Sum ``messages`` into ``num_nodes`` buckets given by ``index``."""
    out = torch.zeros(num_nodes, messages.size(-1), dtype=messages.dtype, device=messages.device)
    out.index_add_(0, index, messages)
    return out


class BaseGNN(nn.Module):
    """Shared interface: ``forward(x, edge_index) -> node embeddings``."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return self.out_channels


class GCN(BaseGNN):
    """Kipf & Welling (2017) graph convolution, with the renormalisation trick.

    ``H^(l+1) = sigma(D~^-1/2 A~ D~^-1/2 H^(l) W^(l))`` with ``A~ = A + I`` and
    ``D~_ii = sum_j A~_ij``, exactly as published.

    The self term therefore enters with weight ``1/(d_i + 1)``, not with weight
    ``1``. Computing ``sigma((A_hat + I) H W)`` with
    ``A_hat = D^-1/2 A D^-1/2`` instead would give the node's own features a
    weight equal to the *entire* rest of its neighbourhood. On a recommender
    graph with mean degree ~50 that would over-weight the self term by ~50x,
    turning this baseline into something much closer to an MLP than to the
    cited model.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers)
        self.dropout = dropout
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # Renormalisation trick: the self term is an edge of A~, so it is
        # normalised by D~ along with every other edge rather than added
        # afterwards at full weight.
        edge_index = add_self_loops(edge_index, num_nodes)
        norm = compute_symmetric_norm(edge_index, num_nodes)
        row, col = edge_index[0], edge_index[1]
        for depth, layer in enumerate(self.layers):
            x = layer(x)
            x = aggregate(x[col] * norm, row, num_nodes)
            if depth < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGE(BaseGNN):
    """Hamilton et al. (2017), mean aggregator.

    ``h_v^(l+1) = sigma(W_self h_v^(l) + W_neigh mean_{u in N(v)} h_u^(l))``
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers)
        self.dropout = dropout
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.self_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.neigh_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        row, col = edge_index[0], edge_index[1]
        degree = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        degree.index_add_(0, row, torch.ones_like(row, dtype=x.dtype))
        degree = degree.clamp(min=1.0).unsqueeze(-1)

        for depth in range(self.num_layers):
            neighbour_mean = aggregate(x[col], row, num_nodes) / degree
            x = self.self_layers[depth](x) + self.neigh_layers[depth](neighbour_mean)
            if depth < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GAT(BaseGNN):
    """Velickovic et al. (2018) graph attention with multi-head attention.

    Attention coefficients use the original additive form,
    ``e_ij = LeakyReLU(a^T [W h_i || W h_j])``, normalised over each node's
    neighbourhood with a numerically stable segment softmax.

    Two details of the paper that are easy to drop and are implemented here:

    * ``h_i' = sigma(sum_{j in N_i} alpha_ij W h_j)`` has ``i in N_i`` -- the
      neighbourhood *includes the node itself*. Self-loops are added inside this
      layer (see :func:`add_self_loops`) rather than in the shared graph builder,
      because LightGCN and NGCF are correct without them.
    * Section 2.1, eq. (6): the final layer **averages** the ``K`` heads,
      ``h_i' = sigma(1/K sum_k sum_j alpha_ij^k W^k h_j)``, rather than running
      a single head. Every layer here uses ``heads`` heads; intermediate layers
      concatenate, the last one averages.
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.2
    ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers)
        self.heads = heads
        self.dropout = dropout

        self.projections = nn.ModuleList()
        self.attn_src = nn.ParameterList()
        self.attn_dst = nn.ParameterList()

        in_dim = in_channels
        for depth in range(num_layers):
            last = depth == num_layers - 1
            out_dim = out_channels if last else hidden_channels
            self.projections.append(nn.Linear(in_dim, out_dim * heads, bias=False))
            self.attn_src.append(nn.Parameter(torch.empty(1, heads, out_dim)))
            self.attn_dst.append(nn.Parameter(torch.empty(1, heads, out_dim)))
            # Intermediate layers concatenate their heads; the last averages them.
            in_dim = out_dim if last else out_dim * heads

        for param in list(self.attn_src) + list(self.attn_dst):
            nn.init.xavier_uniform_(param)

    @staticmethod
    def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Softmax over edges grouped by destination node, max-shifted for stability."""
        max_per_node = torch.full(
            (num_nodes, scores.size(1)), float("-inf"), device=scores.device, dtype=scores.dtype
        )
        max_per_node.scatter_reduce_(
            0, index.unsqueeze(-1).expand_as(scores), scores, reduce="amax", include_self=True
        )
        max_per_node = torch.nan_to_num(max_per_node, neginf=0.0)
        exp_scores = (scores - max_per_node[index]).exp()
        denom = torch.zeros_like(max_per_node).index_add_(0, index, exp_scores)
        return exp_scores / denom[index].clamp(min=1e-16)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # i in N_i: a node attends over its own representation as well as its
        # neighbours', as in the paper and in PyG's GATConv(add_self_loops=True).
        edge_index = add_self_loops(edge_index, num_nodes)
        row, col = edge_index[0], edge_index[1]

        n_heads = self.heads
        for depth in range(self.num_layers):
            last = depth == self.num_layers - 1
            out_dim = self.out_channels if last else self.hidden_channels

            h = self.projections[depth](x).view(num_nodes, n_heads, out_dim)
            alpha_src = (h * self.attn_src[depth]).sum(-1)
            alpha_dst = (h * self.attn_dst[depth]).sum(-1)

            scores = F.leaky_relu(alpha_src[col] + alpha_dst[row], negative_slope=0.2)
            attention = self._segment_softmax(scores, row, num_nodes)
            attention = F.dropout(attention, p=self.dropout, training=self.training)

            messages = h[col] * attention.unsqueeze(-1)
            out = torch.zeros(num_nodes, n_heads, out_dim, device=x.device, dtype=x.dtype)
            out.index_add_(0, row, messages)

            if last:
                # Averaging, not concatenation, in the prediction layer (eq. 6).
                x = out.mean(dim=1)
            else:
                x = out.reshape(num_nodes, n_heads * out_dim)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class NGCF(BaseGNN):
    """Wang et al. (2019) Neural Graph Collaborative Filtering.

    This is the full propagation rule, including the element-wise interaction
    term that distinguishes NGCF from a plain GCN:

    ``m_{u<-i} = 1/sqrt(|N_u||N_i|) * (W1 e_i + W2 (e_i * e_u))``
    ``e_u^(l+1) = LeakyReLU(W1 e_u + sum_{i in N_u} m_{u<-i})``

    The final representation concatenates every layer, as in the paper, then
    projects to ``out_channels`` so the encoder's output dimension is
    comparable across architectures.
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.2
    ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers)
        self.dropout = dropout
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.W1 = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.W2 = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(hidden_channels * (num_layers + 1), out_channels)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        norm = compute_symmetric_norm(edge_index, num_nodes)
        row, col = edge_index[0], edge_index[1]

        h = self.input_projection(x)
        layer_outputs = [h]

        for depth in range(self.num_layers):
            transformed = self.W1[depth](h)
            interaction = self.W2[depth](h[col] * h[row])
            messages = (transformed[col] + interaction) * norm
            aggregated = aggregate(messages, row, num_nodes)

            h = F.leaky_relu(transformed + aggregated, negative_slope=0.2)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.normalize(h, p=2, dim=1)
            layer_outputs.append(h)

        return self.output_projection(torch.cat(layer_outputs, dim=1))


class LightGCN(BaseGNN):
    """He et al. (2020) LightGCN.

    LightGCN removes feature transformation and non-linearity from propagation:
    ``e^(k+1) = sum_{i in N(u)} 1/sqrt(|N_u||N_i|) e_i^(k)``, with the final
    representation a weighted sum over layers.

    One deviation from the paper is documented rather than hidden: LightGCN as
    published learns a free embedding table per node, which cannot generalise to
    nodes unseen at training time. When node features are supplied, a single
    linear projection maps them into embedding space *before* propagation
    begins. Propagation itself remains transformation-free and non-linearity-free,
    so the LightGCN property is preserved; set ``use_input_projection=False``
    with an identity feature matrix to recover the exact published model.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        use_input_projection: bool = True,
        layer_weights: list[float] | None = None,
    ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers)
        self.use_input_projection = use_input_projection
        self.projection: nn.Module
        if use_input_projection:
            self.projection = nn.Linear(in_channels, out_channels)
        else:
            if in_channels != out_channels:
                raise ValueError(
                    "use_input_projection=False requires in_channels == out_channels "
                    f"(got {in_channels} != {out_channels})"
                )
            self.projection = nn.Identity()

        if layer_weights is None:
            weights = [1.0 / (num_layers + 1)] * (num_layers + 1)
        else:
            if len(layer_weights) != num_layers + 1:
                raise ValueError(
                    f"layer_weights must have {num_layers + 1} entries "
                    f"(layers 0..{num_layers}), got {len(layer_weights)}"
                )
            weights = layer_weights
        self.register_buffer("layer_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        norm = compute_symmetric_norm(edge_index, num_nodes)
        row, col = edge_index[0], edge_index[1]

        h = self.projection(x)
        accumulated = self.layer_weights[0] * h

        for depth in range(self.num_layers):
            h = aggregate(h[col] * norm, row, num_nodes)
            accumulated = accumulated + self.layer_weights[depth + 1] * h

        return accumulated


GNN_REGISTRY = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
    "NGCF": NGCF,
    "LightGCN": LightGCN,
}


def build_gnn(name: str, in_channels: int, config) -> BaseGNN:
    """Instantiate a GNN by name using the Phase 1 configuration.

    Raises:
        KeyError: if ``name`` is not a registered architecture. There is no
            silent substitution of one architecture for another.
    """
    if name not in GNN_REGISTRY:
        raise KeyError(
            f"unknown GNN architecture {name!r}; available: {sorted(GNN_REGISTRY)}"
        )

    kwargs = {
        "in_channels": in_channels,
        "hidden_channels": config.hidden_dim,
        "out_channels": config.embedding_dim,
        "num_layers": config.num_layers,
    }
    if name == "GAT":
        kwargs.update(heads=config.gat_heads, dropout=config.dropout)
    elif name in {"GCN", "GraphSAGE", "NGCF"}:
        kwargs.update(dropout=config.dropout)

    return GNN_REGISTRY[name](**kwargs)


def bpr_loss(
    embeddings: torch.Tensor,
    users: torch.Tensor,
    positive_items: torch.Tensor,
    negative_items: torch.Tensor,
) -> torch.Tensor:
    """Bayesian Personalised Ranking loss (Rendle et al., 2009).

    ``-log sigmoid(<e_u, e_i+> - <e_u, e_i->)`` averaged over triples.
    ``logsigmoid`` is used instead of ``log(sigmoid(.) + eps)`` because it is
    numerically stable for large negative margins, where the naive form
    saturates to ``log(eps)`` and stops producing gradient.
    """
    user_emb = embeddings[users]
    positive_scores = (user_emb * embeddings[positive_items]).sum(dim=-1)
    negative_scores = (user_emb * embeddings[negative_items]).sum(dim=-1)
    return -F.logsigmoid(positive_scores - negative_scores).mean()
