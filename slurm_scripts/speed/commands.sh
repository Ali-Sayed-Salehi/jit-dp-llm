#!/encs/bin/tcsh

REPO_ROOT="/speed-scratch/a_s87063/repos/jit-dp-llm"

echo "Preparing training environment"
source /speed-scratch/a_s87063/repos/jit-dp-llm/slurm_scripts/speed/train_prepare.csh


python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/eval_thresholds.py \
--json /speed-scratch/a_s87063/repos/jit-dp-llm/llama/inference/eval_2025-10-06_23-57-06/final_test_results_seq_cls.json \
--out /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/eval_results_2.json \
# -p 0.10 0.20 \



