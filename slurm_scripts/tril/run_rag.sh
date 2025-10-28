#!/bin/bash

#SBATCH --job-name=RAG
#SBATCH --output=/home/alis/links/scratch/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --account=def-pcr
#SBATCH --partition=compute

REPO_ROOT="/home/$USER/links/scratch/repos/jit-dp-llm"

echo "Preparing training environment"
source /home/$USER/links/scratch/repos/jit-dp-llm/slurm_scripts/tril/train_prepare.sh

python rag_main.py build \
  --data ./dataset.jsonl \
  --persist ./rag_index \
  --embedder sentence


# OpenAI
python rag_main.py predict --persist ./rag_index --llm openai  --target "Commit: fix race..." --k 2
# Gemini
python rag_main.py predict --persist ./rag_index --llm gemini  --target "Commit: fix race..." --k 2
# Local HF snapshot
python rag_main.py predict --persist ./rag_index --llm hf-local --hf-model-dir /path/to/snapshot --target "Commit: fix race..." --k 2
