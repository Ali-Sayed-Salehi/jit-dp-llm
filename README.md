# Replication Package for "Risk-Aware Batch Testing for Performance Regression Detection"

This repository is the replication package for the paper **"Risk-Aware Batch Testing for Performance Regression Detection"**. The manuscript is included in this repository as `Batching_for_Performance_Testing.pdf`.

The package supports reproduction of the full paper workflow:

1. construction of the **JIT-Mozilla-Perf** dataset from Mozilla production data sources,
2. fine-tuning of commit-level performance regression risk models,
3. inference to produce chronological commit risk scores, and
4. replay-based CI simulation of risk-aware batching strategies.

This top-level README is intentionally scoped to the paper. The repository may contain other exploratory or legacy materials, but they are not documented here because they are outside the replication package for this paper.

## Paper Summary

The paper studies how machine-learned commit risk can be integrated into Mozilla-style performance testing. Using Firefox as a case study, the replication package combines a production-derived dataset of sheriff-confirmed performance regressions, transformer-based risk predictors, and a realistic CI simulator that models batching, subset testing, backfilling, build delays, and shared worker capacity.

The paper evaluates ModernBERT, CodeBERT, and LLaMA 3.1 8B as commit-level performance regression predictors, then uses their risk scores to drive batching strategies such as Time-Window Batching (TWB), Fixed-Size Batching (FSB), Risk-Adaptive Stream Batching (RASB), Risk-Aged Priority Batching (RAPB), and Risk-Adaptive Trigger Batching (RATB). The main result reported in the paper is that **RAPB-la** provides the strongest overall balance between cost and timeliness, reducing total tests by **32.4%**, reducing maximum time-to-culprit by **26.2%**, and yielding an estimated annual infrastructure savings of about **$491K** relative to the production-inspired baseline.

## Replication Package Contents

The paper has three main technical components, each mapped to concrete repository paths.

- **Paper**: `Batching_for_Performance_Testing.pdf`
- **Dataset construction**: `data_extraction/treeherder/`, `data_extraction/bugzilla/`, `data_extraction/mercurial/`, and `data_extraction/data_preparation.py`
- **Packaged dataset artifacts**: `datasets/mozilla_perf/`
- **Model download, training, and inference**: `llama/`
- **CI simulation and analysis**: `analysis/batch_testing/`
- **Paper-oriented Slurm scripts and commands**: `slurm_scripts/speed/`
- **Containerized environment**: `docker/Dockerfile.llama-train-environment`

## Repository Layout for the Paper

- `datasets/mozilla_perf/`
  The JIT-Mozilla-Perf dataset and derived simulation metadata used throughout the paper.
- `datasets/mozilla_perf/README.md`
  File-level schema and provenance for the dataset artifacts.
- `data_extraction/treeherder/`
  Treeherder and Perfherder extraction code for alerts, failing signatures, signature metadata, per-revision coverage, signature groups, and job durations.
- `data_extraction/bugzilla/`
  Bugzilla extraction code for performance bugs and regressor/regression labeling.
- `data_extraction/mercurial/`
  Mercurial extraction code for Autoland commit history and bug-linked net diffs.
- `data_extraction/data_preparation.py`
  Converts bug-linked diffs into the structured-diff model input used in the paper.
- `llama/train.py`
  Fine-tuning entrypoint for ModernBERT, CodeBERT, and LLaMA-based sequence classification.
- `llama/run_inference.py`
  Inference entrypoint that produces chronological eval/test prediction JSON files consumed by the simulator.
- `llama/configs/templates/train_config_speed_mbert_perf.yaml`
  ModernBERT training template for the Mozilla performance dataset.
- `llama/configs/templates/train_config_speed_codebert_perf.yaml`
  CodeBERT training template for the Mozilla performance dataset.
- `llama/configs/templates/train_config_speed_llama_perf.yaml`
  LLaMA 3.1 8B training template for the Mozilla performance dataset.
- `analysis/batch_testing/simulation.py`
  Main simulator used for Optuna tuning on the eval split and replay on the test split.
- `analysis/batch_testing/model_machine_count.py`
  Worker-capacity sweep used for the infrastructure-scaling analysis.
- `analysis/batch_testing/README.md`
  Detailed documentation of the simulator, batching strategies, bisection strategies, and metrics.
- `slurm_scripts/speed/commands.sh`
  Canonical paper-scale commands used to run the simulator and related analyses on a cluster.

## Main Data and Result Artifacts

### JIT-Mozilla-Perf dataset

The core modeling dataset is stored under `datasets/mozilla_perf/`.

- `perf_llm_struc.jsonl`
  Primary structured-diff dataset used in the paper. It contains **11,384** chronologically ordered commit instances.
- `perf_llm_struc_no_fw_2_6_18.jsonl`
  Post-processed variant that relabels commits after excluding selected Treeherder frameworks. This is the default dataset in the perf-specific training templates shipped in `llama/configs/templates/`.
- `perf_bugs_with_diff.jsonl`
  Bug-linked net diffs extracted from Mozilla Autoland and aligned to Bugzilla performance bugs.
- `alert_summary_fail_perf_sigs.csv`, `all_signatures.jsonl`, `sig_groups.jsonl`, `sig_group_job_durations.csv`, and `perf_jobs_per_revision_details_rectified.jsonl`
  Simulation metadata used to model test selection, signature-group coverage, and job duration.

For the primary paper dataset `perf_llm_struc.jsonl`, the chronological split implemented by the training pipeline is:

- Train: `7399` instances = `7328` clean + `71` regressors
- Eval: `1138` instances = `1126` clean + `12` regressors
- Test: `2847` instances = `2832` clean + `15` regressors

These counts match the split reported in the paper.

### Packaged prediction artifacts

The repository already includes prediction JSON files that can be used directly as simulator inputs.

- `analysis/batch_testing/final_test_results_perf_codebert_eval.json`
- `analysis/batch_testing/final_test_results_perf_codebert_final_test.json`
- `analysis/batch_testing/archive/final_test_results_perf_mbert_eval.json`
- `analysis/batch_testing/archive/final_test_results_perf_mbert_final_test.json`

The CodeBERT final-test prediction file reports a ROC-AUC of `0.6941`, matching the paper's reported CodeBERT test ROC-AUC at the displayed precision.

### Packaged analysis artifacts

- `analysis/batch_testing/results/machine_count_sweep.json`
  Output of the worker-capacity sweep analysis.

## Environment Setup

For a standard Python environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notes:

- GPU-based training and large-model inference require a working PyTorch/CUDA environment.
- Data extraction from Mozilla services requires network access.
- Some extraction steps require credentials in `secrets/.env` using `secrets/.env.example` as a template.
- If you prefer a containerized environment for model work, use `docker/Dockerfile.llama-train-environment`.

## Fastest Reproduction Path

If your goal is to reproduce the paper's **simulation stage** without retraining models, use the packaged CodeBERT prediction files as inputs to the simulator.

```bash
python analysis/batch_testing/simulation.py \
  --input-json-eval analysis/batch_testing/final_test_results_perf_codebert_eval.json \
  --input-json-final analysis/batch_testing/final_test_results_perf_codebert_final_test.json \
  --output-eval analysis/batch_testing/results/batch_eval_mopt.json \
  --output-final analysis/batch_testing/results/batch_test_mopt.json \
  --build-time-minutes 98.7 \
  --mopt-trials 50 \
  --skip-exhaustive-testing \
  --optimize-for-timeliness-metric max_ttc \
  --baseline-opt-metric-multplier 2 \
  --workers-android 60 \
  --workers-windows 120 \
  --workers-linux 100 \
  --workers-mac 250
```

This command reruns the main Optuna-based batch-testing study using the shipped eval/test prediction files. For the full paper-scale command lines, including explicit strategy lists and additional analyses, see `slurm_scripts/speed/commands.sh`.

## Full Regeneration Workflow

The full replication path follows the same order as the paper.

### 1. Rebuild the JIT-Mozilla-Perf dataset

Typical regeneration order:

```bash
python data_extraction/treeherder/get_perf_alerts.py
python data_extraction/bugzilla/get_perf_bugs.py
python data_extraction/mercurial/fetch_all_commit.py
python data_extraction/mercurial/get_bug_diffs.py
python data_extraction/data_preparation.py --mode mozilla_perf_struc
python data_extraction/treeherder/get_failing_perf_sigs.py
python data_extraction/treeherder/get_num_perf_tests.py
python data_extraction/treeherder/get_perf_sigs.py
python data_extraction/treeherder/get_sigs_per_job.py
python data_extraction/treeherder/create_sig_groups.py
python data_extraction/treeherder/get_job_duration.py
python data_extraction/treeherder/rectify_job_count_per_revision.py
python data_extraction/treeherder/filter_perf_llm_struc.py
```

Relevant dataset documentation:

- `datasets/mozilla_perf/README.md`
- `data_extraction/treeherder/README.md`
- `data_extraction/bugzilla/README.md`
- `data_extraction/mercurial/README.md`

### 2. Download base models

Example:

```bash
python llama/download_model.py --model_id answerdotai/ModernBERT-large
python llama/download_model.py --model_id microsoft/codebert-base
python llama/download_model.py --model_id meta-llama/Llama-3.1-8B
```

### 3. Fine-tune the risk predictors

Examples:

```bash
accelerate launch --config_file llama/configs/accelerate_config_deepspeed.yaml \
  llama/train.py --config llama/configs/templates/train_config_speed_mbert_perf.yaml

accelerate launch --config_file llama/configs/accelerate_config_deepspeed.yaml \
  llama/train.py --config llama/configs/templates/train_config_speed_codebert_perf.yaml

accelerate launch --config_file llama/configs/accelerate_config_deepspeed.yaml \
  llama/train.py --config llama/configs/templates/train_config_speed_llama_perf.yaml
```

### 4. Run inference to generate simulator inputs

Run once on the eval split and once on the final test split for each model checkpoint. The eval split is selected with `--eval_ds_as_final_test`.

Example shape of the command:

```bash
python llama/run_inference.py \
  --base_model_path LLMs/snapshots/microsoft/codebert-base \
  --model_path <checkpoint_dir> \
  --dataset_path datasets/mozilla_perf/perf_llm_struc_no_fw_2_6_18.jsonl \
  --mixed_precision bf16

python llama/run_inference.py \
  --base_model_path LLMs/snapshots/microsoft/codebert-base \
  --model_path <checkpoint_dir> \
  --dataset_path datasets/mozilla_perf/perf_llm_struc_no_fw_2_6_18.jsonl \
  --mixed_precision bf16 \
  --eval_ds_as_final_test
```

### 5. Generate historical repeat counts for HATS-style policies

This step is needed for the history-aware subset-suite strategies used in the simulator.

```bash
python data_extraction/treeherder/find_historical_risk_scores.py \
  --eval-preds-json analysis/batch_testing/final_test_results_perf_codebert_eval.json
```

### 6. Run the simulator

```bash
python analysis/batch_testing/simulation.py --help
python analysis/batch_testing/model_machine_count.py --help
```

Use the generated eval/test prediction JSON files as `--input-json-eval` and `--input-json-final`.

## What the Simulator Produces

For each batching and bisection configuration, the simulator reports:

- total tests executed,
- total CPU time,
- mean feedback time,
- mean time-to-culprit,
- maximum time-to-culprit,
- percentile TTC metrics,
- number of true regressors found, and
- feasibility with respect to detecting all regressors.

These are the metrics used in the paper's cost-latency comparisons and Pareto analysis.

## Citation

If you use this replication package, please cite the paper and the software artifacts in this repository. See `CITATION.cff`.
