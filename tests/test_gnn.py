"""Tests for the graph encoders."""

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 (conventional torch alias)

from gcrl.config import GNNConfig
from gcrl.data.graphs import to_undirected
from gcrl.models.gnn import (
    GAT,
    GCN,
    GNN_REGISTRY,
    LightGCN,
    add_self_loops,
    aggregate,
    bpr_loss,
    build_gnn,
    compute_symmetric_norm,
)


@pytest.fixture
def graph():
    torch.manual_seed(0)
    n_nodes, n_edges, n_features = 50, 200, 16
    return (
        torch.randn(n_nodes, n_features),
        torch.randint(0, n_nodes, (2, n_edges)),
        n_nodes,
    )


class TestArchitectures:
    @pytest.mark.parametrize("name", sorted(GNN_REGISTRY))
    def test_forward_produces_finite_embeddings(self, name, graph):
        x, edge_index, n_nodes = graph
        model = build_gnn(name, x.size(1), GNNConfig(embedding_dim=8, hidden_dim=16, num_layers=2))
        out = model(x, edge_index)
        assert out.shape == (n_nodes, 8)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("name", sorted(GNN_REGISTRY))
    def test_gradients_reach_every_parameter(self, name, graph):
        x, edge_index, _ = graph
        model = build_gnn(name, x.size(1), GNNConfig(embedding_dim=8, hidden_dim=16, num_layers=2))
        model(x, edge_index).sum().backward()
        without_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not without_grad, f"{name}: no gradient for {without_grad}"

    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_configured_depth_is_honoured(self, depth, graph):
        """The paper reports a layer count; the model must actually use it."""
        x, edge_index, _ = graph
        model = build_gnn("GCN", x.size(1), GNNConfig(embedding_dim=8, num_layers=depth))
        assert model.num_layers == depth
        assert len(model.layers) == depth

    def test_ngcf_actually_uses_the_graph(self, graph):
        """NGCF was previously two Linear layers that ignored edge_index."""
        x, edge_index, n_nodes = graph
        model = build_gnn("NGCF", x.size(1), GNNConfig(embedding_dim=8, hidden_dim=16, num_layers=2))
        model.eval()
        with torch.no_grad():
            with_edges = model(x, edge_index)
            without_edges = model(x, torch.zeros((2, 0), dtype=torch.long))
        assert not torch.allclose(with_edges, without_edges), "NGCF output ignores the topology"

    def test_every_architecture_uses_the_graph(self, graph):
        x, edge_index, _ = graph
        for name in GNN_REGISTRY:
            model = build_gnn(name, x.size(1), GNNConfig(embedding_dim=8, hidden_dim=16, num_layers=2))
            model.eval()
            with torch.no_grad():
                a = model(x, edge_index)
                b = model(x, torch.zeros((2, 0), dtype=torch.long))
            assert not torch.allclose(a, b), f"{name} output does not depend on edges"

    def test_unknown_architecture_raises(self):
        with pytest.raises(KeyError, match="unknown GNN architecture"):
            build_gnn("NotAGNN", 8, GNNConfig())


class TestLightGCN:
    def test_exact_mode_requires_matching_dimensions(self):
        from gcrl.models.gnn import LightGCN

        with pytest.raises(ValueError, match="requires in_channels == out_channels"):
            LightGCN(16, 32, 8, num_layers=2, use_input_projection=False)

    def test_rejects_wrong_number_of_layer_weights(self):
        from gcrl.models.gnn import LightGCN

        with pytest.raises(ValueError, match="layer_weights must have"):
            LightGCN(8, 16, 8, num_layers=3, layer_weights=[0.5, 0.5])


class TestPrimitives:
    def test_symmetric_norm_handles_isolated_nodes(self):
        edge_index = torch.tensor([[0, 1], [1, 0]])
        norm = compute_symmetric_norm(edge_index, num_nodes=5)
        assert torch.isfinite(norm).all()

    def test_aggregate_sums_into_the_right_buckets(self):
        messages = torch.tensor([[1.0], [2.0], [3.0]])
        index = torch.tensor([0, 0, 1])
        out = aggregate(messages, index, num_nodes=3)
        torch.testing.assert_close(out, torch.tensor([[3.0], [3.0], [0.0]]))


class TestBPRLoss:
    def test_is_lower_when_positives_score_higher(self):
        embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        users, positives, negatives = torch.tensor([0]), torch.tensor([1]), torch.tensor([2])
        good = bpr_loss(embeddings, users, positives, negatives)
        bad = bpr_loss(embeddings, users, negatives, positives)
        assert good < bad

    def test_is_stable_for_large_negative_margins(self):
        """log(sigmoid(x) + eps) saturates and stops producing gradient; logsigmoid does not."""
        embeddings = torch.tensor([[100.0], [-100.0], [100.0]], requires_grad=True)
        loss = bpr_loss(embeddings, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(embeddings.grad).all()
        assert embeddings.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Reference implementations, derived from the papers rather than from the code
# under test. Nothing below reads an expected value out of gcrl.models.gnn --
# only the learned weights are borrowed, so the assertions compare two
# independent computations of the same published equation.
# ---------------------------------------------------------------------------


def _undirected(pairs):
    """Symmetric edge_index from a list of ``(src, dst)`` pairs."""
    edges = torch.tensor(pairs, dtype=torch.long).T
    return torch.unique(torch.cat([edges, edges.flip(0)], dim=1), dim=1)


def _circulant(degree, n):
    """A ``degree``-regular circulant graph on ``n`` nodes (``degree`` even)."""
    pairs = [(i, (i + k) % n) for i in range(n) for k in range(1, degree // 2 + 1)]
    return _undirected(pairs)


def _kipf_dense(x, edge_index, num_nodes, weight, bias):
    """Kipf & Welling (2017) eq. 2, built densely: ``D~^-1/2 A~ D~^-1/2 X W``.

    ``A~ = A + I`` and ``D~_ii = sum_j A~_ij``, so the self term is normalised
    together with every neighbour rather than added afterwards at weight 1.
    """
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    for src, dst in edge_index.T.tolist():
        if src != dst:
            adjacency[src, dst] = 1.0
    a_tilde = adjacency + torch.eye(num_nodes, dtype=torch.float64)
    d_inv_sqrt = torch.diag(a_tilde.sum(dim=1).pow(-0.5))
    transformed = x.double() @ weight.double().T + bias.double()
    return d_inv_sqrt @ a_tilde @ d_inv_sqrt @ transformed


def _gcn_propagation_matrix(num_nodes, edge_index):
    """Recover ``S`` from a one-layer GCN by setting ``W = I`` and ``b = 0``."""
    model = GCN(num_nodes, num_nodes, num_nodes, num_layers=1, dropout=0.0)
    model.eval()
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.eye(num_nodes))
        model.layers[0].bias.zero_()
        return model(torch.eye(num_nodes), edge_index)


def _gat_paper_reference(model, x, edge_index, num_nodes, *, self_loops=True, heads_at_output=None):
    """Velickovic et al. (2018) eqs. (1)-(6), dense and loop-based.

    Neighbourhoods are assembled as explicit python sets and the softmax is
    taken over that set, so the result depends on the paper's definition of
    ``N_i`` rather than on how the module happens to index its edges.

    Args:
        self_loops: whether ``i in N_i`` (the paper says it is).
        heads_at_output: number of heads the final layer uses; ``None`` means
            all ``K`` of them, averaged as in eq. (6).
    """
    n_heads = model.heads
    current = x.double()
    for depth in range(model.num_layers):
        last = depth == model.num_layers - 1
        out_dim = model.out_channels if last else model.hidden_channels
        all_weights = model.projections[depth].weight.double()
        a_src = model.attn_src[depth].double().squeeze(0)
        a_dst = model.attn_dst[depth].double().squeeze(0)

        neighbours = {i: set() for i in range(num_nodes)}
        for src, dst in edge_index.T.tolist():
            if src != dst:
                neighbours[src].add(dst)
        if self_loops:
            for i in range(num_nodes):
                neighbours[i].add(i)

        used = n_heads if (not last or heads_at_output is None) else heads_at_output
        per_head = []
        for k in range(used):
            w_k = all_weights[k * out_dim : (k + 1) * out_dim]
            h = current @ w_k.T
            out = torch.zeros(num_nodes, out_dim, dtype=torch.float64)
            for i in range(num_nodes):
                js = sorted(neighbours[i])
                if not js:
                    continue
                scores = torch.tensor(
                    [
                        F.leaky_relu(a_dst[k] @ h[i] + a_src[k] @ h[j], negative_slope=0.2)
                        for j in js
                    ],
                    dtype=torch.float64,
                )
                alpha = (scores - scores.max()).exp()
                alpha = alpha / alpha.sum()
                for weight_ij, j in zip(alpha, js, strict=True):
                    out[i] = out[i] + weight_ij * h[j]
            per_head.append(out)

        if last:
            current = torch.stack(per_head).mean(dim=0) if used > 1 else per_head[0]
        else:
            current = F.elu(torch.cat(per_head, dim=1))
    return current


class TestGCNRenormalisationTrick:
    """DEFECT 1: the self term must be normalised, not added at weight 1.

    The pre-fix code computed ``sigma((A_hat + I) X W)`` with
    ``A_hat = D^-1/2 A D^-1/2``, giving the node's own features weight 1 while
    each neighbour got ``1/sqrt(d_i d_j)``. Kipf & Welling put the self-loop
    *inside* ``A~`` so it is normalised with everything else.
    """

    def test_two_node_graph_matches_the_hand_computed_matrix(self):
        """A single edge 0--1. By hand: A~ = [[1,1],[1,1]], D~ = diag(2,2),
        so S = D~^-1/2 A~ D~^-1/2 = [[0.5, 0.5], [0.5, 0.5]].

        Adding the self-loop a second time would give [[1, 1], [1, 1]], every
        entry twice too large.
        """
        edge_index = _undirected([(0, 1)])
        propagation = _gcn_propagation_matrix(2, edge_index)
        expected = torch.full((2, 2), 0.5)
        torch.testing.assert_close(propagation, expected, rtol=1e-5, atol=1e-6)

    def test_matches_a_dense_kipf_reference_on_a_bipartite_graph(self):
        """4-node bipartite graph, checked against the dense published form."""
        torch.manual_seed(0)
        num_nodes = 4
        edge_index = _undirected([(0, 2), (0, 3), (1, 2)])
        x = torch.randn(num_nodes, 3)

        model = GCN(3, 5, 2, num_layers=1, dropout=0.0)
        model.eval()
        with torch.no_grad():
            got = model(x, edge_index).double()
        expected = _kipf_dense(x, edge_index, num_nodes, model.layers[0].weight,
                               model.layers[0].bias)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(("degree", "n"), [(2, 5), (4, 9), (6, 13)])
    def test_neighbour_to_self_mass_is_d_to_one_on_a_regular_graph(self, degree, n):
        """On a d-regular graph Kipf gives every entry of A~ weight 1/(d+1),
        so the d neighbours carry d times the node's own mass.

        Putting weight 1 on self and 1/d on each of d neighbours would give a
        1:1 split at every degree, which is not the published normalisation.
        """
        edge_index = _circulant(degree, n)
        degrees = torch.zeros(n).index_add_(
            0, edge_index[0], torch.ones(edge_index.size(1))
        )
        assert (degrees == degree).all(), f"graph is not {degree}-regular"

        propagation = _gcn_propagation_matrix(n, edge_index)
        self_mass = propagation[0, 0].item()
        neighbour_mass = (propagation[0].sum() - propagation[0, 0]).item()

        assert self_mass == pytest.approx(1.0 / (degree + 1), rel=1e-5)
        assert neighbour_mass / self_mass == pytest.approx(float(degree), rel=1e-4)

    def test_self_loops_are_idempotent(self):
        """GCN adds A~ = A + I itself, so handing it a graph that already has
        self-loops must not double-count them."""
        torch.manual_seed(1)
        num_nodes = 6
        edge_index = _undirected([(0, 3), (0, 4), (1, 3), (2, 5)])
        x = torch.randn(num_nodes, 4)
        model = GCN(4, 5, 3, num_layers=2, dropout=0.0)
        model.eval()
        with torch.no_grad():
            plain = model(x, edge_index)
            with_loops = model(x, add_self_loops(edge_index, num_nodes))
        torch.testing.assert_close(plain, with_loops)


class TestAddSelfLoops:
    def test_adds_exactly_one_loop_per_node(self):
        edge_index = _undirected([(0, 1), (1, 2)])
        out = add_self_loops(edge_index, 4)
        loops = out[:, out[0] == out[1]]
        assert torch.equal(torch.bincount(loops[0], minlength=4), torch.ones(4, dtype=torch.long))

    def test_pre_existing_loops_are_not_counted_twice(self):
        edge_index = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 2, 2, 0]], dtype=torch.long)
        out = add_self_loops(edge_index, 3)
        loops = out[:, out[0] == out[1]]
        assert torch.equal(torch.bincount(loops[0], minlength=3), torch.ones(3, dtype=torch.long))

    def test_keeps_every_non_loop_edge(self):
        edge_index = _undirected([(0, 1), (1, 2)])
        out = add_self_loops(edge_index, 3)
        kept = {tuple(e) for e in out.T.tolist() if e[0] != e[1]}
        assert kept == {tuple(e) for e in edge_index.T.tolist()}


class TestGATSelfLoopsAndHeads:
    """DEFECT 2: ``i in N_i``, and the output layer averages K heads.

    Pre-fix the layer aggregated over strict neighbours only, and the final
    layer used ``n_heads = 1``.
    """

    @pytest.fixture
    def model_and_graph(self):
        torch.manual_seed(1)
        num_nodes = 6
        edge_index = _undirected([(0, 3), (0, 4), (1, 3), (2, 5)])
        x = torch.randn(num_nodes, 4)
        model = GAT(4, 5, 3, num_layers=2, heads=4, dropout=0.0)
        model.eval()
        return model, x, edge_index, num_nodes

    def test_matches_the_paper_reference(self, model_and_graph):
        model, x, edge_index, num_nodes = model_and_graph
        with torch.no_grad():
            got = model(x, edge_index).double()
        expected = _gat_paper_reference(model, x, edge_index, num_nodes)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    def test_differs_from_a_strict_neighbour_reference(self, model_and_graph):
        """Guards the assertion above: dropping ``i in N_i`` changes the answer,
        so matching the paper reference is not vacuous."""
        model, x, edge_index, num_nodes = model_and_graph
        with torch.no_grad():
            got = model(x, edge_index).double()
        no_loops = _gat_paper_reference(model, x, edge_index, num_nodes, self_loops=False)
        assert (got - no_loops).abs().max() > 1e-3

    def test_differs_from_a_single_head_output_layer(self, model_and_graph):
        """Likewise for eq. (6): a 1-head prediction layer is a different model."""
        model, x, edge_index, num_nodes = model_and_graph
        with torch.no_grad():
            got = model(x, edge_index).double()
        one_head = _gat_paper_reference(model, x, edge_index, num_nodes, heads_at_output=1)
        assert (got - one_head).abs().max() > 1e-3

    def test_output_layer_carries_every_head(self, model_and_graph):
        """Structural counterpart: the last layer must parameterise K heads."""
        model, _, _, _ = model_and_graph
        assert model.projections[-1].out_features == model.heads * model.out_channels
        assert model.attn_src[-1].shape == (1, model.heads, model.out_channels)
        assert model.attn_dst[-1].shape == (1, model.heads, model.out_channels)

    def test_isolated_node_attends_to_itself(self):
        """Hand-derived: with ``N_i = {i}`` the softmax is degenerate, alpha_ii = 1,
        so the output is ``mean_k W^k h_i``. Without self-loops it is exactly 0."""
        torch.manual_seed(2)
        edge_index = _undirected([(0, 1)])  # node 2 has no neighbours
        x = torch.randn(3, 4)
        model = GAT(4, 5, 3, num_layers=1, heads=2, dropout=0.0)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        expected = (x[2] @ model.projections[0].weight.T).view(2, 3).mean(dim=0)
        torch.testing.assert_close(out[2], expected, rtol=1e-5, atol=1e-6)
        assert out[2].norm() > 1e-3, "isolated node collapsed to zero: self-loop missing"

    def test_attention_over_a_two_node_graph_is_hand_computable(self):
        """One edge, one head, scalar features: alpha is a two-term softmax."""
        import math

        model = GAT(1, 1, 1, num_layers=1, heads=1, dropout=0.0)
        model.eval()
        with torch.no_grad():
            model.projections[0].weight.fill_(2.0)
            model.attn_src[0].fill_(0.5)
            model.attn_dst[0].fill_(-0.25)
            x = torch.tensor([[1.0], [3.0]])
            out = model(x, _undirected([(0, 1)]))

        h0, h1 = 2.0, 6.0  # W h_i
        def leaky(v):
            return v if v > 0 else 0.2 * v
        # node 0 attends over N_0 = {0, 1}
        e00 = leaky(-0.25 * h0 + 0.5 * h0)
        e01 = leaky(-0.25 * h0 + 0.5 * h1)
        w00, w01 = math.exp(e00), math.exp(e01)
        expected0 = (w00 * h0 + w01 * h1) / (w00 + w01)
        assert out[0].item() == pytest.approx(expected0, rel=1e-5)


class TestSelfLoopScope:
    """The self-loop fix must stay inside GCN and GAT.

    LightGCN's rule is pure neighbourhood aggregation with no self-connection,
    NGCF carries its self term as ``W1 e_u``, and GraphSAGE has a dedicated
    ``W_self``. Adding self-loops to the shared graph would corrupt all three.
    """

    def test_shared_graph_builder_adds_no_self_loops(self):
        edge_index = to_undirected(torch.tensor([[0, 0, 1], [2, 3, 2]], dtype=torch.long))
        assert not bool((edge_index[0] == edge_index[1]).any())

    def test_lightgcn_propagation_has_an_exactly_zero_diagonal(self):
        """He et al. (2020): e_u^(k+1) sums over N(u) only, so S_uu = 0."""
        num_nodes = 6
        edge_index = _undirected([(0, 3), (0, 4), (1, 3), (2, 5)])
        model = LightGCN(
            num_nodes, num_nodes, num_nodes, num_layers=1,
            use_input_projection=False, layer_weights=[0.0, 1.0],
        )
        model.eval()
        with torch.no_grad():
            propagation = model(torch.eye(num_nodes), edge_index)
        assert propagation.diag().abs().max().item() == 0.0

    @pytest.mark.parametrize("name", ["GraphSAGE", "LightGCN", "NGCF"])
    def test_other_architectures_do_not_add_self_loops(self, name):
        """If they added loops internally, feeding a graph that already has them
        would leave the output unchanged, as it does for GCN and GAT."""
        torch.manual_seed(3)
        num_nodes = 8
        edge_index = _undirected([(0, 4), (0, 5), (1, 4), (2, 6), (3, 7)])
        x = torch.randn(num_nodes, 4)
        config = GNNConfig(embedding_dim=5, hidden_dim=6, num_layers=2, dropout=0.0)
        model = build_gnn(name, 4, config)
        model.eval()
        with torch.no_grad():
            plain = model(x, edge_index)
            with_loops = model(x, add_self_loops(edge_index, num_nodes))
        assert (plain - with_loops).abs().max() > 1e-4, (
            f"{name} ignores input self-loops, so it is adding its own"
        )

    @pytest.mark.parametrize("name", ["GAT", "GCN"])
    def test_gcn_and_gat_do_add_self_loops(self, name):
        torch.manual_seed(3)
        num_nodes = 8
        edge_index = _undirected([(0, 4), (0, 5), (1, 4), (2, 6), (3, 7)])
        x = torch.randn(num_nodes, 4)
        config = GNNConfig(embedding_dim=5, hidden_dim=6, num_layers=2, dropout=0.0)
        model = build_gnn(name, 4, config)
        model.eval()
        with torch.no_grad():
            plain = model(x, edge_index)
            with_loops = model(x, add_self_loops(edge_index, num_nodes))
        torch.testing.assert_close(plain, with_loops)
