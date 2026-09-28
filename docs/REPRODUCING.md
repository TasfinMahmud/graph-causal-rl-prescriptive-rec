# Reproducing the results

All reported results are produced by the commands below under seed 42. Runtimes
are measured on the hardware listed at the end of this document.

## 1. Environment

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The GPU used is an NVIDIA RTX 5060 Ti, which is Blackwell (`sm_120`). **CUDA
12.4 wheels have no kernels for it and fail at the first matmul while
`torch.cuda.is_available()` still returns `True`.** Install a cu128 or later
build and verify with a real multiplication rather than the availability flag:

```bash
python -c "import torch; x=torch.randn(8,8,device='cuda'); print((x@x).sum().item())"
```

Then confirm the test suite passes without any data present:

```bash
pytest -q          # the full suite, no data required
```

## 2. Datasets

Download and place the datasets as follows. Neither is redistributed here.

```
code/data/raw/obd/           Open Bandit Dataset, BTS subset  (~6.3 GB)
code/data/raw/kuairec/       KuaiRec 2.0                      (~3.5 GB)
```

- Open Bandit Dataset: <https://research.zozo.com/data.html>
- KuaiRec 2.0: <https://kuairec.com/>

The loaders read `big_matrix.csv` and `small_matrix.csv` for KuaiRec and the
BTS `all` split for OBD. `configs/obd.yaml` and `configs/kuairec.yaml` document
what each file is used for and why.

## 3. Running the experiments

Each experiment is four phases: graph representation, causal estimation, policy
optimization, off-policy evaluation. `run_paper_experiments.py` chains all four.

```bash
# OBD, main arm: all four phases, about 9.75 h end to end
python scripts/run_paper_experiments.py --config configs/obd.yaml \
       --variants main --seeds 42

# OBD ablations
python scripts/run_paper_experiments.py --config configs/obd.yaml \
       --variants ablation_raw ablation_cate --seeds 42

# KuaiRec, main arm: Phase 3 is about 8.5 h, Phase 4 about 18 min
python -m gcrl.cli phase1 --config configs/_generated_kuairec_main.yaml --seed 42
python -m gcrl.cli phase3 --config configs/_generated_kuairec_main.yaml --seed 42
python -m gcrl.cli phase4 --config configs/_generated_kuairec_main.yaml --seed 42

# The estimator-validation run: reuses the policies above, no retraining
python -m gcrl.cli phase1 --config configs/kuairec_validate_ope_full.yaml --seed 42
python -m gcrl.cli phase4 --config configs/kuairec_validate_ope_full.yaml --seed 42
```

`scripts/estimate_runtime.py` times a short probe of each phase on the current
machine and extrapolates to a chosen configuration, which is useful before
committing to a full run.

Measured Phase 3 times on OBD (1,599,990 rounds, 80 actions): IQL 87 min, DQN
125 min, CQL 148 min, BCQ 165 min. KuaiRec (1,480,517 rounds, 3,327 actions) is
comparable per agent.

## 4. Diagnostics and reference policies

The reference policies, action-support counts, permutation tests and the
duration correlation all come from one script. It runs on the CPU by design, so
it can be run beside a training job:

```bash
python scripts/reference_policies.py --config configs/_generated_kuairec_main.yaml --seed 42
python scripts/reference_policies.py --config configs/_generated_kuairec_ablation_raw.yaml --seed 42
python scripts/reference_policies.py --config configs/kuairec_sage_bandits.yaml --seed 42
```

## 5. Tables and figures

```bash
python scripts/generate_paper_tables.py --results-dir results
python scripts/make_figures.py --results-dir results --out-dir docs/figures
```

The first command writes the LaTeX tables and the claims check into
`results/paper/`. The second regenerates both figures, as PDF and PNG, from the
same result files.

## Known limitations

**GAT and NGCF exceed the memory available on a 16 GiB GPU for the KuaiRec
graph.** Both architectures materialise one message per edge. Over 3,088,460
edges, a single (edges x 128) float32 tensor occupies 1.47 GiB, placing GAT near
36 GiB and NGCF near 27 GiB across three layers once the backward pass retains
them. GAT raises `OutOfMemoryError`; NGCF spills into shared system memory under
Windows and does not complete an epoch. `configs/kuairec_encoders.yaml`
therefore trains the three architectures that fit within the available memory.
Both are retained on the Open Bandit Dataset, whose graph is substantially
smaller.

**The estimator-validation run subsamples its evaluation block** to 250,000
rounds. DM, DR and MRDR fit a reward model on `[context, one-hot(action)]`; over
all 4,676,570 rounds, float64 promotion within scikit-learn makes that design
matrix 94.5 GiB. All 1,411 users are retained, so the cost is a reward model
fitted on fewer (user, item, reward) triples. This is documented in
`configs/kuairec_validate_ope_full.yaml`.

## Hardware

AMD Ryzen 7 5700G (8C/16T), 32 GB RAM, NVIDIA RTX 5060 Ti 16 GB, Windows,
Python 3.10, PyTorch 2.x with CUDA 12.8.
