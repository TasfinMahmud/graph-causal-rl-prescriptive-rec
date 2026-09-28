"""Configuration loading and validation tests."""

import pytest

from gcrl.config import ConfigError, load_config, resolve_device


def _write(tmp_path, body, name="c.yaml"):
    path = tmp_path / name
    path.write_text(body)
    return path


class TestValidation:
    def test_rejects_unknown_keys(self, tmp_path):
        """A typo must fail loudly, not be silently ignored."""
        path = _write(tmp_path, "gnn:\n  epochz: 50\n")
        with pytest.raises(ConfigError, match="unknown keys"):
            load_config(path)

    def test_rejects_split_ratios_that_do_not_sum_to_one(self, tmp_path):
        path = _write(tmp_path, "dataset:\n  name: d\n  loader: obd\nsplit:\n  train_ratio: 0.8\n  val_ratio: 0.3\n  test_ratio: 0.1\n")
        with pytest.raises(ConfigError, match="must sum to 1.0"):
            load_config(path)

    def test_rejects_identical_treatment_and_outcome(self, tmp_path):
        path = _write(tmp_path, "dataset:\n  name: d\n  loader: obd\ncausal:\n  enabled: true\n  treatment_column: y\n  outcome_column: y\n")
        with pytest.raises(ConfigError, match="identical"):
            load_config(path)

    def test_rejects_cate_shaping_without_causal_phase(self, tmp_path):
        """lambda > 0 with causal disabled would compare a run against itself."""
        path = _write(tmp_path, "dataset:\n  name: d\n  loader: obd\ncausal:\n  enabled: false\nrl:\n  cate_reward_weight: 0.5\n")
        with pytest.raises(ConfigError, match="requires causal.enabled"):
            load_config(path)

    def test_rejects_rl_architecture_not_trained_in_phase1(self, tmp_path):
        path = _write(tmp_path, "dataset:\n  name: d\n  loader: obd\ngnn:\n  architectures: [GCN]\nrl:\n  state_source: gnn_embeddings\n  gnn_architecture: LightGCN\n")
        with pytest.raises(ConfigError, match="not trained in"):
            load_config(path)

    def test_rejects_unknown_ope_estimator(self, tmp_path):
        path = _write(tmp_path, "dataset:\n  name: d\n  loader: obd\nope:\n  estimators: [magic]\n")
        with pytest.raises(ConfigError, match="unknown OPE estimators"):
            load_config(path)

    def test_rejects_invalid_expectile(self, tmp_path):
        path = _write(tmp_path, "dataset:\n  name: d\n  loader: obd\nrl:\n  iql_expectile: 1.5\n")
        with pytest.raises(ConfigError, match="iql_expectile"):
            load_config(path)


class TestInheritance:
    def test_child_overrides_parent(self, tmp_path):
        _write(tmp_path, "seed: 1\ngnn:\n  epochs: 50\n  embedding_dim: 64\n", "base.yaml")
        child = _write(tmp_path, "extends: base.yaml\ndataset:\n  name: d\n  loader: obd\ngnn:\n  epochs: 5\n", "child.yaml")
        config = load_config(child)
        assert config.gnn.epochs == 5
        assert config.gnn.embedding_dim == 64  # inherited
        assert config.seed == 1


class TestShippedConfigs:
    @pytest.mark.parametrize("name", ["obd", "kuairec"])
    def test_shipped_configs_are_valid(self, name):
        config = load_config(f"configs/{name}.yaml")
        assert config.dataset.name == name

    def test_kuairec_disables_causal_phase(self):
        """KuaiRec has no randomised treatment, so uplift must be off."""
        assert load_config("configs/kuairec.yaml").causal.enabled is False

    def test_obd_uses_logged_propensities(self):
        config = load_config("configs/obd.yaml")
        assert config.ope.use_logged_propensity
        assert "snipw" in config.ope.estimators


class TestDeviceResolution:
    """Device resolution, asserted on BOTH branches regardless of the host.

    Reading ``torch.cuda.is_available()`` and asserting whatever the current
    machine happens to be would make these tests silently hardware dependent:
    on a CPU-only host every assertion would take the ``cpu`` branch and pass,
    so a regression in the CUDA branch could not fail them there and would
    surface only on a machine with a real card. Both branches are therefore
    forced.
    """

    @staticmethod
    def _cuda(available: bool):
        import unittest.mock as mock

        import torch

        return mock.patch.object(torch.cuda, "is_available", lambda: available)

    def test_auto_resolves_to_cpu_without_a_gpu(self):
        with self._cuda(False):
            assert resolve_device("auto") == "cpu"

    def test_auto_resolves_to_an_indexed_device_with_a_gpu(self):
        with self._cuda(True):
            assert resolve_device("auto") == "cuda:0"

    def test_cuda_falls_back_to_cpu_when_unavailable(self):
        """A CUDA request on a machine without one must degrade, not crash.

        The original concern -- that hardcoding a CUDA device made the code
        unrunnable without a GPU -- still holds and is still tested here. What
        changed is only the *name* of the device when one IS present: an
        indexed ``cuda:0`` rather than a bare ``cuda``, because d3rlpy splits
        that string on ``:`` when it reloads a checkpoint.
        """
        with self._cuda(False):
            assert resolve_device("cuda") == "cpu"
            assert resolve_device("cuda:1") == "cpu"

    def test_cuda_request_is_honoured_when_available(self):
        with self._cuda(True):
            assert resolve_device("cuda") == "cuda:0"

    def test_explicit_cpu_is_respected(self):
        for available in (True, False):
            with self._cuda(available):
                assert resolve_device("cpu") == "cpu"


class TestCudaDeviceCarriesAnIndex:
    """A CUDA device must always be resolved WITH an index, never bare "cuda".

    d3rlpy's ``torch_utility.map_location`` does ``_, index = device.split(":")``
    when it reloads a checkpoint, so a bare ``"cuda"`` raises
    ``ValueError: not enough values to unpack (expected 2, got 1)``.

    Phase 3 trains and saves without ever touching that path, so the failure
    surfaces only in Phase 4 -- after the expensive work is done -- and its
    message names neither the device nor d3rlpy. This exact defect cost a full
    smoke-test cycle on real data, which is why it is pinned here.
    """

    @staticmethod
    def _with_cuda(available: bool):
        import contextlib
        import unittest.mock as mock

        import torch

        return mock.patch.object(torch.cuda, "is_available", lambda: available) \
            if True else contextlib.nullcontext()

    def test_auto_resolves_to_an_indexed_cuda_device(self):
        from gcrl.config import resolve_device

        with self._with_cuda(True):
            assert resolve_device("auto") == "cuda:0"

    def test_bare_cuda_request_gains_an_index(self):
        from gcrl.config import resolve_device

        with self._with_cuda(True):
            assert resolve_device("cuda") == "cuda:0"

    def test_an_explicit_index_is_preserved(self):
        """cuda:1 must keep addressing GPU 1, not be rewritten to GPU 0."""
        from gcrl.config import resolve_device

        with self._with_cuda(True):
            assert resolve_device("cuda:1") == "cuda:1"

    def test_cpu_is_left_alone(self):
        from gcrl.config import resolve_device

        with self._with_cuda(True):
            assert resolve_device("cpu") == "cpu"
        with self._with_cuda(False):
            assert resolve_device("auto") == "cpu"

    def test_every_resolved_cuda_device_survives_the_d3rlpy_split(self):
        """The precise operation that crashed, asserted directly."""
        from gcrl.config import resolve_device

        with self._with_cuda(True):
            for requested in ("auto", "cuda", "cuda:0", "cuda:1"):
                resolved = resolve_device(requested)
                assert resolved.startswith("cuda")
                _, index = resolved.split(":")   # d3rlpy does exactly this
                assert index.isdigit(), (requested, resolved)

    def test_d3rlpy_policy_normalises_a_bare_cuda_device(self):
        """Even a caller that bypasses resolve_device must not reach d3rlpy bare."""
        from gcrl.models.rl import D3RLPyPolicy

        policy = D3RLPyPolicy("DQN", n_actions=3, context_dim=4, device="cuda")
        assert policy.device == "cuda:0"
        _, index = policy.device.split(":")
        assert index == "0"
