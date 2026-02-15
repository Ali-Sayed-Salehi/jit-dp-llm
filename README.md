# jit-dp-llm

Research code for investigating how to use LLMs for just-in-time defect prediction (JIT-DP), plus simulators for “debugging policies” (batch testing / git bisect) on Mozilla/Apache datasets.

## How things fit together (high level)

1. Build (or download) datasets into `datasets/` (via `data_extraction/`).
2. Train/run models to produce per-commit risk scores (via `llama/` or `openai/`).
3. Evaluate downstream policies and metrics using the simulators in `analysis/`.

## Repository layout (start here)

- `analysis/`: experiment analysis + simulators (offline once datasets/predictions are present)
  - `analysis/batch_testing/`: simulator for **batch testing** + follow-up bisection policies on the Mozilla performance regression dataset. Entrypoint: `analysis/batch_testing/simulation.py`. Docs: `analysis/batch_testing/README.md`.
  - `analysis/git_bisect/`: simulator for **git-bisect-style** debugging on the Mozilla JIT bug dataset (lookback + bisection strategies). Entrypoint: `analysis/git_bisect/simulate.py`. Docs: `analysis/git_bisect/README.md`.
  - `analysis/jit_dp/`: small analysis helpers (e.g., threshold selection from model outputs, token-length plots).
- `data_extraction/`: scripts to build datasets from upstream systems (Bugzilla, Treeherder, Mercurial, GitHub, etc.). Each subfolder has its own README.
- `datasets/`: extracted dataset artifacts used by training/inference/analysis (tracked via DVC as `datasets.dvc`). Start with `datasets/README.md`.
- `llama/`: local model workflow (download weights, finetune, merge LoRA adapters, run inference, plot metrics).
- `openai/`: API-based inference utilities (OpenAI / Gemini).
- `metrics/`: custom metrics for Hugging Face `evaluate`.
- `docker/`: containerized environment + helper scripts.
- `slurm_scripts/`: Slurm submission scripts (cluster-specific).
- `secrets/`: `.env` template for tokens/keys.

## Setup

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows (PowerShell)
# .\\venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

# Optional (if your GPU + toolchain supports it)
pip install flash-attn --no-build-isolation
```

Create `secrets/.env` (copy from `secrets/.env.example`) and fill out the keys you need.

## Models and data

This repo is designed to run offline once you have:
- local model weights (download via `python llama/download_model.py`)
- dataset files under `datasets/`

### Getting datasets

- Download a prepared snapshot:
  - https://drive.google.com/drive/folders/1QsMAn7yboltMN7gJmXDu-qKIplsjJyKq?usp=sharing
- Or pull via DVC (tracks `datasets/` as `datasets.dvc`):
  - `pip install "dvc[gdrive]"`
  - `dvc pull datasets.dvc`
- Or re-extract from sources using `data_extraction/` (see its subfolder READMEs). If using Hugging Face datasets, `python llama/fetch_dataset.py` downloads them to local disk first.

### Dataset docs

- `datasets/mozilla_perf/README.md`
- `datasets/mozilla_jit/README.md`
- `datasets/apachejit/README.md`

## Running the `analysis/` simulators

- Batch testing simulator:
  - `python analysis/batch_testing/simulation.py --help`
  - Docs: `analysis/batch_testing/README.md`
- Git bisect simulator:
  - `python analysis/git_bisect/simulate.py --help`
  - Docs: `analysis/git_bisect/README.md`

## Training / inference

- Create `llama/configs/train_config_local.yaml` from `llama/configs/train_config.yaml`, then run:
  - `python llama/train.py --config llama/configs/train_config_local.yaml`
- For larger runs, use `slurm_scripts/` or `docker/` (see `docker/` scripts).

## Citation

See `CITATION.cff`.
