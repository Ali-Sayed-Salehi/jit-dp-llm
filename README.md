# Replication Package for “Look Back Before You Bisect: A Risk-Aware Approach for Efficient Bisection”

This repository is the replication package for the paper *Look Back Before You Bisect: A
Risk-Aware Approach for Efficient Bisection*. It contains the artifacts needed to reproduce the
paper's workflow around:

- constructing the **MozillaJIT** dataset from Mozilla Bugzilla and Autoland history
- converting MozillaJIT into the structured commit representation used for risk modeling
- training and running commit-level risk models on MozillaJIT
- simulating **lookback + bisection** culprit-localization policies using those risk scores
- storing the paper's precomputed risk predictions and frozen simulation outputs

The MozillaJIT dataset used by this replication package is archived separately on Zenodo:
[https://doi.org/10.5281/zenodo.18829451](https://doi.org/10.5281/zenodo.18829451)

This repository's name, `jit-dp-llm`, is historical and broader than the paper. For this Zenodo
release, the relevant subset of the repository is the MozillaJIT extraction pipeline, the risk
model training and inference workflow, and the `analysis/git_bisect/` simulation artifacts. This
README intentionally focuses only on those paper-related components.

## What This Package Contains

This replication package is organized around four linked artifacts.

### 1. MozillaJIT dataset and extraction pipeline

The paper introduces **MozillaJIT**, a Mozilla-specific just-in-time defect prediction dataset in
which each example represents a Bugzilla bug together with the net code change that landed for that
bug in Mozilla's `autoland` Mercurial repository.

Relevant paths:

- `datasets/mozilla_jit/`
- `data_extraction/bugzilla/get_all_bugs.py`
- `data_extraction/mercurial/fetch_all_commit.py`
- `data_extraction/mercurial/link_bug_diffs.py`
- `data_extraction/data_preparation.py`
- `data_extraction/utils.py`

At a high level, the extraction pipeline:

1. pulls Bugzilla bugs and regression links from Mozilla's Bugzilla instance
2. exports ordered commit metadata from Mozilla's `autoland` Mercurial history
3. links bugs to contiguous `Bug <id>` landing blocks and computes a single net diff per bug
4. converts the joined bug+diff dataset into the structured prompt format used for risk modeling

The standalone MozillaJIT dataset release is:
[https://doi.org/10.5281/zenodo.18829451](https://doi.org/10.5281/zenodo.18829451)

### 2. Risk-model training and inference code

The paper trains commit-level risk models on MozillaJIT and compares **ModernBERT** and
**LLaMA-3.1 8B** sequence-classification models using commit messages plus structured diffs.

Relevant paths:

- `llama/train.py`
- `llama/run_inference.py`
- `llama/configs/templates/train_config_speed_mbert_mozilla_jit.yaml`
- `llama/configs/templates/train_config_speed_llama_mozilla_jit.yaml`

In the paper workflow, these components are used to:

- train sequence classifiers on `datasets/mozilla_jit/jit_llm_struc_2022.jsonl`
- evaluate them on chronological validation and final-test splits
- generate per-commit risk scores that are later consumed by the simulation code

The precomputed risk prediction files included in this repository are under `analysis/git_bisect/`
and are the direct simulator inputs used for the paper's end-to-end localization experiments.

### 3. Lookback and bisection simulator

The paper's main evaluation is implemented in `analysis/git_bisect/`. This simulator models
culprit localization as two coupled phases:

- **lookback**, which searches backward from a failing observation to find a clean boundary and
  reduce the candidate interval
- **bisection**, which searches inside the resulting interval to isolate the culprit commit

Relevant paths:

- `analysis/git_bisect/simulate.py`
- `analysis/git_bisect/lookback.py`
- `analysis/git_bisect/bisection.py`
- `analysis/git_bisect/README.md`

The simulator consumes:

- the joined MozillaJIT bug dataset
- the ordered Autoland commit stream
- the precomputed risk prediction JSON files

It produces the aggregate metrics reported in the paper, including total localization steps,
mean localization cost, and worst-case localization cost across strategy combinations.

### 4. Frozen paper outputs

The repository includes the precomputed outputs used for the paper's simulation results.

Relevant paths:

- `analysis/git_bisect/risk_predictions_eval.json`
- `analysis/git_bisect/risk_predictions_final_test.json`
- `analysis/git_bisect/final_experiment_results/simulation_optuna_eval.json`
- `analysis/git_bisect/final_experiment_results/simulation_optuna_final_test.json`
- `analysis/git_bisect/final_experiment_results/pareto_front_stats.json`
- `analysis/git_bisect/final_experiment_results/pareto_front_distributions.png`

The directory `analysis/git_bisect/results/` is the default output location for reruns. In the
current repository snapshot, the JSON outputs in `analysis/git_bisect/results/` and
`analysis/git_bisect/final_experiment_results/` are identical.

## Paper-to-Code Terminology

The paper uses descriptive algorithm names, while the simulator often uses shorter class names or
compact strategy codes in JSON outputs. The following mapping is the main source of naming
differences.

- **MozillaJIT** in the paper corresponds to the dataset under `datasets/mozilla_jit/`.
- The paper's **StandardMidpointBisection** corresponds to the simulator's `GitBisectBaseline`
  class and the short strategy code `GB`.
- The paper's **NoLookback** corresponds to the simulator strategy code `NLB`.
- The paper's **NoLookback + StandardMidpointBisection** baseline appears in the result JSON files
  as `NLB+GB`.
- The paper's **culprit localization steps** correspond to the simulator's test-count metrics:
  `total_tests`, `total_lookback_tests`, `total_bisection_tests`,
  `mean_tests_per_search`, and `max_tests_per_search`.
- The paper's **evaluation split** corresponds to the code's `eval` split and to files such as
  `risk_predictions_eval.json` and `simulation_optuna_eval.json`.
- The paper's **final test split** corresponds to the code's `final_test` split and to files such
  as `risk_predictions_final_test.json` and `simulation_optuna_final_test.json`.
- The paper's **RiskWeightedLookbackSum** and **RiskWeightedLookbackLogSurvival** appear in result
  files with abbreviated strategy codes such as `RWLBS` and `RWLBLS`.
- The paper's **RiskWeightedBisectionSum** and **RiskWeightedBisectionLogSurvival** appear in
  result files as `RWBS` and `RWBLS`.
- Adaptive and constrained variants are encoded with suffixes in the simulator:
  `-AD` for adaptive decrease, `-AI` for adaptive increase, and `-FF` for forced fallback.
- The paper refers to the dataset generically as MozillaJIT, but the snapshot files included in
  this repository use historical filenames such as `mozilla_jit_2022.jsonl` and
  `jit_llm_struc_2022.jsonl`.
- The simulator's default bug path is `datasets/mozilla_jit/mozilla_jit.jsonl`, while the
  archived snapshot included in this repository is named `datasets/mozilla_jit/mozilla_jit_2022.jsonl`.
  For reruns, either pass `--bugs-path datasets/mozilla_jit/mozilla_jit_2022.jsonl` explicitly or
  create a copy/symlink named `mozilla_jit.jsonl`.
- Likewise, `data_extraction/data_preparation.py --mode mozilla_jit_struc` currently writes
  `jit_llm_struc_2025.jsonl` by default, while the paper snapshot included here is
  `jit_llm_struc_2022.jsonl`.

## Relevant Repository Layout

Only the following parts of the repository are needed for the paper's replication workflow:

- `datasets/mozilla_jit/`
  - MozillaJIT artifacts and dataset documentation
- `data_extraction/bugzilla/`
  - Bugzilla collection scripts for MozillaJIT
- `data_extraction/mercurial/`
  - Autoland commit export and bug-to-diff linking scripts
- `data_extraction/data_preparation.py`
  - conversion from the joined MozillaJIT dataset to the structured LLM-ready format
- `data_extraction/utils.py`
  - structured diff conversion and commit-message cleanup utilities
- `llama/`
  - model training and inference code used for the risk-model experiments
- `analysis/git_bisect/`
  - lookback strategies, bisection strategies, simulator, precomputed predictions, and paper
    results
- `requirements.txt`
  - Python dependencies for the extraction, training, inference, and simulation workflow

## Quick Start: Reproduce the Paper's Simulation Results

This is the shortest path if you want to reproduce the main end-to-end simulation using the
precomputed risk predictions already included in the repository.

### 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Obtain the MozillaJIT dataset

Download the MozillaJIT dataset from Zenodo and place the files under `datasets/mozilla_jit/`:

[https://doi.org/10.5281/zenodo.18829451](https://doi.org/10.5281/zenodo.18829451)

The simulator can be pointed directly at the archived bug file name
`datasets/mozilla_jit/mozilla_jit_2022.jsonl`, so you do not need to rename the Zenodo snapshot to
run the paper experiments.

### 3. Run the simulator with the included risk predictions

```bash
python analysis/git_bisect/simulate.py \
  --bugs-path datasets/mozilla_jit/mozilla_jit_2022.jsonl \
  --commits-path datasets/mozilla_jit/all_commits.jsonl \
  --risk-eval analysis/git_bisect/risk_predictions_eval.json \
  --risk-final analysis/git_bisect/risk_predictions_final_test.json \
  --output-eval analysis/git_bisect/results/simulation_optuna_eval.json \
  --output-final analysis/git_bisect/results/simulation_optuna_final_test.json
```

This command:

- tunes strategy parameters on the chronological evaluation split
- replays the selected parameters on the chronological final-test split
- writes rerun outputs into `analysis/git_bisect/results/`

### 4. Compare rerun outputs to the frozen paper outputs

Compare the generated JSON files in `analysis/git_bisect/results/` to the frozen paper copies in
`analysis/git_bisect/final_experiment_results/`.

The key output files are:

- `analysis/git_bisect/results/simulation_optuna_eval.json`
- `analysis/git_bisect/results/simulation_optuna_final_test.json`
- `analysis/git_bisect/results/pareto_front_stats.json`
- `analysis/git_bisect/results/pareto_front_distributions.png`

The corresponding frozen paper artifacts are in:

- `analysis/git_bisect/final_experiment_results/`

## Full Workflow: Regenerate the Pipeline End to End

If you want to regenerate the paper workflow rather than only replay the frozen artifacts, the
steps are:

### 1. Re-extract MozillaJIT from upstream sources

```bash
python data_extraction/bugzilla/get_all_bugs.py
python data_extraction/mercurial/fetch_all_commit.py
python data_extraction/mercurial/link_bug_diffs.py
python data_extraction/data_preparation.py --mode mozilla_jit_struc
```

Important note:

- `fetch_all_commit.py` writes `datasets/mozilla_perf/all_commits.jsonl` by default, while the
  MozillaJIT linker expects `datasets/mozilla_jit/all_commits.jsonl`. Copy or symlink the exported
  commit file into `datasets/mozilla_jit/` before running `link_bug_diffs.py`, or override the
  script paths explicitly.

### 2. Train the risk models

Use the MozillaJIT-specific config templates:

- `llama/configs/templates/train_config_speed_llama_mozilla_jit.yaml`
- `llama/configs/templates/train_config_speed_mbert_mozilla_jit.yaml`

Example command pattern:

```bash
python llama/train.py --config llama/configs/templates/train_config_speed_llama_mozilla_jit.yaml
```

The same pattern applies to the ModernBERT config by replacing the config path.

### 3. Generate risk predictions

Run inference on the trained checkpoint using `llama/run_inference.py`. The resulting JSON output is
the simulator input format used by `analysis/git_bisect/simulate.py`.

Example command pattern:

```bash
python llama/run_inference.py \
  --model_path /path/to/checkpoint \
  --base_model_path /path/to/base/model \
  --dataset_path datasets/mozilla_jit/jit_llm_struc_2022.jsonl \
  --output_dir /path/to/inference_output
```

The paper workflow uses separate prediction files for the chronological evaluation and final-test
splits, stored in this repository as:

- `analysis/git_bisect/risk_predictions_eval.json`
- `analysis/git_bisect/risk_predictions_final_test.json`

### 4. Run the lookback+bisection simulation

After generating the risk prediction JSON files, run `analysis/git_bisect/simulate.py` as shown in
the quick-start section.

## Included Main Result Files

The main simulation outputs bundled with this package are:

- `analysis/git_bisect/final_experiment_results/simulation_optuna_eval.json`
- `analysis/git_bisect/final_experiment_results/simulation_optuna_final_test.json`

These files contain one row per strategy combination and record:

- the strategy code pair, for example `NLB+GB` or `RWLBS-AI+RWBS`
- total localization cost
- mean localization cost per processed bug
- worst-case localization cost per processed bug
- how many culprits were successfully localized
- the selected hyperparameters for tuned strategies

## Replication Package Description

This repository is the replication package for *Look Back Before You Bisect: A Risk-Aware Approach
for Efficient Bisection*. It contains the code and artifacts needed to reproduce the paper's Mozilla
history-based workflow for risk-aware culprit localization. The package covers four connected parts
of the study: construction of the MozillaJIT dataset from Mozilla Bugzilla and Autoland history,
conversion of MozillaJIT into the structured commit representation used for learning, training and
inference of commit-level risk models on MozillaJIT, and simulation of lookback-plus-bisection
localization strategies using those risk scores. The standalone MozillaJIT dataset used by this
package is archived separately at [https://doi.org/10.5281/zenodo.18829451](https://doi.org/10.5281/zenodo.18829451).

Within the repository, the MozillaJIT extraction pipeline is implemented in
`data_extraction/bugzilla/`, `data_extraction/mercurial/`, `data_extraction/data_preparation.py`,
and `data_extraction/utils.py`. These scripts collect Bugzilla bugs, export ordered Autoland commit
metadata, link regressions to landed code changes, compute one net diff per bug, and generate the
structured XML-like diff representation used by the models. The risk-model part of the paper is
implemented in `llama/train.py`, `llama/run_inference.py`, and the MozillaJIT-specific config
templates under `llama/configs/templates/`, which support the ModernBERT and LLaMA-3.1 8B
experiments described in the paper. The simulation framework used for the paper's end-to-end
evaluation is implemented in `analysis/git_bisect/`, including lookback strategies, bisection
strategies, the main simulator entrypoint, the precomputed risk prediction files for the evaluation
and final-test splits, and frozen copies of the paper's final result JSON files and Pareto-front
artifacts.

The repository name `jit-dp-llm` is historical, and several code paths use short internal names
that differ from the terminology used in the paper. In particular, the paper's
StandardMidpointBisection appears in the code as `GitBisectBaseline` with strategy code `GB`, and
the paper's NoLookback + StandardMidpointBisection baseline appears in result files as `NLB+GB`.
Likewise, the paper's localization-step metric appears in the simulator outputs as test-count
metrics such as `total_tests`, `mean_tests_per_search`, and `max_tests_per_search`. The archived
MozillaJIT snapshot bundled for experiments also uses historical filenames such as
`mozilla_jit_2022.jsonl` and `jit_llm_struc_2022.jsonl`, while some scripts still default to the
generic names `mozilla_jit.jsonl` and `jit_llm_struc_2025.jsonl`. This README documents those
name differences so the replication workflow can be followed without ambiguity.
