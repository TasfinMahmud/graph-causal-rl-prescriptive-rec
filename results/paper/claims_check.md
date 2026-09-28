# Claims check — KUAIREC — experiment `kuairec`

Seeds are pooled only within this one experiment. Numbers from another experiment on the same dataset are reported separately below, never averaged in.

## Exact

| Rank | Agent | Value | Seeds used / attempted | Complete? |
|---|---|---|---|---|
| 1 | NeuralUCB | 0.70930 | 1 / 1 | yes |
| 2 | BCQ | 0.15515 | 1 / 1 | yes |
| 3 | IQL | 0.09944 | 1 / 1 | yes |
| 4 | LinUCB | 0.05540 | 1 / 1 | yes |
| 5 | DQN | 0.05030 | 1 / 1 | yes |
| 6 | Random | 0.04645 | 1 / 1 | yes |
| 7 | CQL | 0.02704 | 1 / 1 | yes |

**Best under Exact: NeuralUCB** (0.70930).

- Claim *"BCQ outperforms unconstrained DQN"*: SUPPORTED (BCQ=0.15515 vs DQN=0.05030).
- Claim *"CQL outperforms unconstrained DQN"*: **NOT SUPPORTED** (CQL=0.02704 vs DQN=0.05030).
- Agents beating the random baseline (0.04645): LinUCB, NeuralUCB, IQL, DQN, BCQ.
- Agents **not** beating random: CQL. Any claim of general improvement must exclude these.

## Ablation: graph state vs raw features (Exact)

- BCQ: GNN=0.15515 vs raw=0.29809 -> **graph does not help**
- CQL: GNN=0.02704 vs raw=0.00142 -> graph helps
- DQN: GNN=0.05030 vs raw=0.29809 -> **graph does not help**
- IQL: GNN=0.09944 vs raw=0.69882 -> **graph does not help**
- LinUCB: GNN=0.05540 vs raw=0.04738 -> graph helps
- NeuralUCB: GNN=0.70930 vs raw=0.03862 -> graph helps
- Random: GNN=0.04645 vs raw=0.04645 -> **graph does not help**


# Claims check — KUAIREC — experiment `kuairec_validate_ope_full`

Seeds are pooled only within this one experiment. Numbers from another experiment on the same dataset are reported separately below, never averaged in.

## DM

| Rank | Agent | Value | Seeds used / attempted | Complete? |
|---|---|---|---|---|
| 1 | NeuralUCB | 0.71383 | 1 / 1 | yes |
| 2 | BCQ | 0.12574 | 1 / 1 | yes |
| 3 | IQL | 0.08618 | 1 / 1 | yes |
| 4 | Random | 0.04787 | 1 / 1 | yes |
| 5 | DQN | 0.04772 | 1 / 1 | yes |
| 6 | CQL | 0.04770 | 1 / 1 | yes |
| 7 | LinUCB | 0.04619 | 1 / 1 | yes |

**Best under DM: NeuralUCB** (0.71383).

- Claim *"BCQ outperforms unconstrained DQN"*: SUPPORTED (BCQ=0.12574 vs DQN=0.04772).
- Claim *"CQL outperforms unconstrained DQN"*: **NOT SUPPORTED** (CQL=0.04770 vs DQN=0.04772).
- Agents beating the random baseline (0.04787): NeuralUCB, IQL, BCQ.
- Agents **not** beating random: LinUCB, DQN, CQL. Any claim of general improvement must exclude these.

## DR

| Rank | Agent | Value | Seeds used / attempted | Complete? |
|---|---|---|---|---|
| 1 | NeuralUCB | 0.71867 | 1 / 1 | yes |
| 2 | BCQ | 0.12688 | 1 / 1 | yes |
| 3 | IQL | 0.08866 | 1 / 1 | yes |
| 4 | Random | 0.04779 | 1 / 1 | yes |
| 5 | DQN | 0.04630 | 1 / 1 | yes |
| 6 | CQL | 0.04576 | 1 / 1 | yes |
| 7 | LinUCB | 0.04515 | 1 / 1 | yes |

**Best under DR: NeuralUCB** (0.71867).

- Claim *"BCQ outperforms unconstrained DQN"*: SUPPORTED (BCQ=0.12688 vs DQN=0.04630).
- Claim *"CQL outperforms unconstrained DQN"*: **NOT SUPPORTED** (CQL=0.04576 vs DQN=0.04630).
- Agents beating the random baseline (0.04779): NeuralUCB, IQL, BCQ.
- Agents **not** beating random: LinUCB, DQN, CQL. Any claim of general improvement must exclude these.
- **Caveat:** these agents' importance-weighted estimates rest on very few rounds and must be reported with this caveat — BCQ: ESS 67.5 of 250,000 rounds (0.027%); CQL: ESS 68.7 of 250,000 rounds (0.027%); DQN: ESS 61.4 of 250,000 rounds (0.025%); IQL: ESS 79.6 of 250,000 rounds (0.032%); LinUCB: ESS 56.7 of 250,000 rounds (0.023%); NeuralUCB: ESS 75.6 of 250,000 rounds (0.030%).

## Exact

| Rank | Agent | Value | Seeds used / attempted | Complete? |
|---|---|---|---|---|
| 1 | NeuralUCB | 0.70843 | 1 / 1 | yes |
| 2 | BCQ | 0.15505 | 1 / 1 | yes |
| 3 | IQL | 0.09904 | 1 / 1 | yes |
| 4 | LinUCB | 0.05500 | 1 / 1 | yes |
| 5 | DQN | 0.05118 | 1 / 1 | yes |
| 6 | Random | 0.04647 | 1 / 1 | yes |
| 7 | CQL | 0.02682 | 1 / 1 | yes |

**Best under Exact: NeuralUCB** (0.70843).

- Claim *"BCQ outperforms unconstrained DQN"*: SUPPORTED (BCQ=0.15505 vs DQN=0.05118).
- Claim *"CQL outperforms unconstrained DQN"*: **NOT SUPPORTED** (CQL=0.02682 vs DQN=0.05118).
- Agents beating the random baseline (0.04647): LinUCB, NeuralUCB, IQL, DQN, BCQ.
- Agents **not** beating random: CQL. Any claim of general improvement must exclude these.

## MRDR

| Rank | Agent | Value | Seeds used / attempted | Complete? |
|---|---|---|---|---|
| 1 | NeuralUCB | 0.75221 | 1 / 1 | yes |
| 2 | BCQ | 0.14940 | 1 / 1 | yes |
| 3 | IQL | 0.14276 | 1 / 1 | yes |
| 4 | Random | 0.04719 | 1 / 1 | yes |
| 5 | LinUCB | 0.03471 | 1 / 1 | yes |
| 6 | DQN | 0.02126 | 1 / 1 | yes |
| 7 | CQL | 0.01367 | 1 / 1 | yes |

**Best under MRDR: NeuralUCB** (0.75221).

- Claim *"BCQ outperforms unconstrained DQN"*: SUPPORTED (BCQ=0.14940 vs DQN=0.02126).
- Claim *"CQL outperforms unconstrained DQN"*: **NOT SUPPORTED** (CQL=0.01367 vs DQN=0.02126).
- Agents beating the random baseline (0.04719): NeuralUCB, IQL, BCQ.
- Agents **not** beating random: LinUCB, DQN, CQL. Any claim of general improvement must exclude these.
- **Caveat:** these agents' importance-weighted estimates rest on very few rounds and must be reported with this caveat — BCQ: ESS 67.5 of 250,000 rounds (0.027%); CQL: ESS 68.7 of 250,000 rounds (0.027%); DQN: ESS 61.4 of 250,000 rounds (0.025%); IQL: ESS 79.6 of 250,000 rounds (0.032%); LinUCB: ESS 56.7 of 250,000 rounds (0.023%); NeuralUCB: ESS 75.6 of 250,000 rounds (0.030%).

## SNIPW

| Rank | Agent | Value | Seeds used / attempted | Complete? |
|---|---|---|---|---|
| 1 | NeuralUCB | 0.80532 | 1 / 1 | yes |
| 2 | BCQ | 0.14062 | 1 / 1 | yes |
| 3 | IQL | 0.12680 | 1 / 1 | yes |
| 4 | Random | 0.04638 | 1 / 1 | yes |
| 5 | DQN | 0.01677 | 1 / 1 | yes |
| 6 | LinUCB | 0.01611 | 1 / 1 | yes |
| 7 | CQL | 0.01426 | 1 / 1 | yes |

**Best under SNIPW: NeuralUCB** (0.80532).

- Claim *"BCQ outperforms unconstrained DQN"*: SUPPORTED (BCQ=0.14062 vs DQN=0.01677).
- Claim *"CQL outperforms unconstrained DQN"*: **NOT SUPPORTED** (CQL=0.01426 vs DQN=0.01677).
- Agents beating the random baseline (0.04638): NeuralUCB, IQL, BCQ.
- Agents **not** beating random: LinUCB, DQN, CQL. Any claim of general improvement must exclude these.
- **Caveat:** these agents' importance-weighted estimates rest on very few rounds and must be reported with this caveat — BCQ: ESS 67.5 of 250,000 rounds (0.027%); CQL: ESS 68.7 of 250,000 rounds (0.027%); DQN: ESS 61.4 of 250,000 rounds (0.025%); IQL: ESS 79.6 of 250,000 rounds (0.032%); LinUCB: ESS 56.7 of 250,000 rounds (0.023%); NeuralUCB: ESS 75.6 of 250,000 rounds (0.030%).


# Claims check — OBD — experiment `obd`

Seeds are pooled only within this one experiment. Numbers from another experiment on the same dataset are reported separately below, never averaged in.

## DR

| Rank | Agent | Value | Seeds used / attempted | Complete? |
|---|---|---|---|---|
| 1 | DQN | 0.01741 | 1 / 1 | yes |
| 2 | CQL | 0.00627 | 1 / 1 | yes |
| 3 | IQL | 0.00580 | 1 / 1 | yes |
| 4 | Random | 0.00406 | 1 / 1 | yes |
| 5 | LinUCB | 0.00285 | 1 / 1 | yes |
| 6 | NeuralUCB | 0.00169 | 1 / 1 | yes |
| 7 | BCQ | 0.00128 | 1 / 1 | yes |

**Best under DR: DQN** (0.01741).

- Claim *"BCQ outperforms unconstrained DQN"*: **NOT SUPPORTED** (BCQ=0.00128 vs DQN=0.01741).
- Claim *"CQL outperforms unconstrained DQN"*: **NOT SUPPORTED** (CQL=0.00627 vs DQN=0.01741).
- Agents beating the random baseline (0.00406): IQL, DQN, CQL.
- Agents **not** beating random: LinUCB, NeuralUCB, BCQ. Any claim of general improvement must exclude these.
- **Caveat:** these agents' importance-weighted estimates rest on very few rounds and must be reported with this caveat — BCQ: ESS 117.8 of 200,000 rounds (0.059%); DQN: ESS 53.8 of 200,000 rounds (0.027%); LinUCB: ESS 2.9 of 200,000 rounds (0.001%); NeuralUCB: ESS 39.3 of 200,000 rounds (0.020%); Random: ESS 1818.8 of 200,000 rounds (0.909%).

## MRDR

| Rank | Agent | Value | Seeds used / attempted | Complete? |
|---|---|---|---|---|
| 1 | DQN | 0.02466 | 1 / 1 | yes |
| 2 | CQL | 0.00616 | 1 / 1 | yes |
| 3 | IQL | 0.00602 | 1 / 1 | yes |
| 4 | Random | 0.00403 | 1 / 1 | yes |
| 5 | BCQ | 0.00139 | 1 / 1 | yes |
| 6 | NeuralUCB | 0.00139 | 1 / 1 | yes |
| 7 | LinUCB | 0.00000 | 1 / 1 | yes |

**Best under MRDR: DQN** (0.02466).

- Claim *"BCQ outperforms unconstrained DQN"*: **NOT SUPPORTED** (BCQ=0.00139 vs DQN=0.02466).
- Claim *"CQL outperforms unconstrained DQN"*: **NOT SUPPORTED** (CQL=0.00616 vs DQN=0.02466).
- Agents beating the random baseline (0.00403): IQL, DQN, CQL.
- Agents **not** beating random: LinUCB, NeuralUCB, BCQ. Any claim of general improvement must exclude these.
- **Caveat:** these agents' importance-weighted estimates rest on very few rounds and must be reported with this caveat — BCQ: ESS 117.8 of 200,000 rounds (0.059%); DQN: ESS 53.8 of 200,000 rounds (0.027%); LinUCB: ESS 2.9 of 200,000 rounds (0.001%); NeuralUCB: ESS 39.3 of 200,000 rounds (0.020%); Random: ESS 1818.8 of 200,000 rounds (0.909%).

## SNIPW

| Rank | Agent | Value | Seeds used / attempted | Complete? |
|---|---|---|---|---|
| 1 | DQN | 0.01496 | 1 / 1 | yes |
| 2 | CQL | 0.00631 | 1 / 1 | yes |
| 3 | IQL | 0.00578 | 1 / 1 | yes |
| 4 | Random | 0.00408 | 1 / 1 | yes |
| 5 | NeuralUCB | 0.00149 | 1 / 1 | yes |
| 6 | BCQ | 0.00146 | 1 / 1 | yes |
| 7 | LinUCB | 0.00000 | 1 / 1 | yes |

**Best under SNIPW: DQN** (0.01496).

- Claim *"BCQ outperforms unconstrained DQN"*: **NOT SUPPORTED** (BCQ=0.00146 vs DQN=0.01496).
- Claim *"CQL outperforms unconstrained DQN"*: **NOT SUPPORTED** (CQL=0.00631 vs DQN=0.01496).
- Agents beating the random baseline (0.00408): IQL, DQN, CQL.
- Agents **not** beating random: LinUCB, NeuralUCB, BCQ. Any claim of general improvement must exclude these.
- **Caveat:** these agents' importance-weighted estimates rest on very few rounds and must be reported with this caveat — BCQ: ESS 117.8 of 200,000 rounds (0.059%); DQN: ESS 53.8 of 200,000 rounds (0.027%); LinUCB: ESS 2.9 of 200,000 rounds (0.001%); NeuralUCB: ESS 39.3 of 200,000 rounds (0.020%); Random: ESS 1818.8 of 200,000 rounds (0.909%).

## Ablation: graph state vs raw features (DR)

- BCQ: GNN=0.00128 vs raw=0.00459 -> **graph does not help**
- CQL: GNN=0.00627 vs raw=0.00573 -> graph helps
- DQN: GNN=0.01741 vs raw=0.02568 -> **graph does not help**
- IQL: GNN=0.00580 vs raw=0.00624 -> **graph does not help**
- LinUCB: GNN=0.00285 vs raw=0.00276 -> graph helps
- NeuralUCB: GNN=0.00169 vs raw=0.00998 -> **graph does not help**
- Random: GNN=0.00406 vs raw=0.00415 -> **graph does not help**

## Ablation: CATE reward shaping (DR)

- BCQ: lambda=0 0.00128 vs shaped 0.00062 -> **shaping does not help**
- CQL: lambda=0 0.00627 vs shaped 0.00626 -> **shaping does not help**
- DQN: lambda=0 0.01741 vs shaped 0.01500 -> **shaping does not help**
- IQL: lambda=0 0.00580 vs shaped 0.00579 -> **shaping does not help**
- LinUCB: lambda=0 0.00285 vs shaped 0.00282 -> **shaping does not help**
- NeuralUCB: lambda=0 0.00169 vs shaped -0.00054 -> **shaping does not help**
- Random: lambda=0 0.00406 vs shaped 0.00406 -> **shaping does not help**

## Ablation: graph state vs raw features (MRDR)

- BCQ: GNN=0.00139 vs raw=0.00419 -> **graph does not help**
- CQL: GNN=0.00616 vs raw=0.00564 -> graph helps
- DQN: GNN=0.02466 vs raw=0.01636 -> graph helps
- IQL: GNN=0.00602 vs raw=0.00637 -> **graph does not help**
- LinUCB: GNN=0.00000 vs raw=0.00000 -> **graph does not help**
- NeuralUCB: GNN=0.00139 vs raw=0.00709 -> **graph does not help**
- Random: GNN=0.00403 vs raw=0.00427 -> **graph does not help**

## Ablation: CATE reward shaping (MRDR)

- BCQ: lambda=0 0.00139 vs shaped 0.00004 -> **shaping does not help**
- CQL: lambda=0 0.00616 vs shaped 0.00617 -> shaping helps
- DQN: lambda=0 0.02466 vs shaped 0.01167 -> **shaping does not help**
- IQL: lambda=0 0.00602 vs shaped 0.00601 -> **shaping does not help**
- LinUCB: lambda=0 0.00000 vs shaped 0.00000 -> **shaping does not help**
- NeuralUCB: lambda=0 0.00139 vs shaped 0.00000 -> **shaping does not help**
- Random: lambda=0 0.00403 vs shaped 0.00403 -> **shaping does not help**

## Ablation: graph state vs raw features (SNIPW)

- BCQ: GNN=0.00146 vs raw=0.00410 -> **graph does not help**
- CQL: GNN=0.00631 vs raw=0.00578 -> graph helps
- DQN: GNN=0.01496 vs raw=0.02048 -> **graph does not help**
- IQL: GNN=0.00578 vs raw=0.00619 -> **graph does not help**
- LinUCB: GNN=0.00000 vs raw=0.00000 -> **graph does not help**
- NeuralUCB: GNN=0.00149 vs raw=0.00906 -> **graph does not help**
- Random: GNN=0.00408 vs raw=0.00408 -> **graph does not help**

## Ablation: CATE reward shaping (SNIPW)

- BCQ: lambda=0 0.00146 vs shaped 0.00004 -> **shaping does not help**
- CQL: lambda=0 0.00631 vs shaped 0.00631 -> **shaping does not help**
- DQN: lambda=0 0.01496 vs shaped 0.01231 -> **shaping does not help**
- IQL: lambda=0 0.00578 vs shaped 0.00577 -> **shaping does not help**
- LinUCB: lambda=0 0.00000 vs shaped 0.00000 -> **shaping does not help**
- NeuralUCB: lambda=0 0.00149 vs shaped 0.00000 -> **shaping does not help**
- Random: lambda=0 0.00408 vs shaped 0.00408 -> **shaping does not help**
