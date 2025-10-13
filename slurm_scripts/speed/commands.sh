#!/encs/bin/tcsh

REPO_ROOT="/speed-scratch/a_s87063/repos/jit-dp-llm"

echo "Preparing training environment"
source /speed-scratch/a_s87063/repos/jit-dp-llm/slurm_scripts/speed/train_prepare.csh


# python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/eval_thresholds.py \
# --json /speed-scratch/a_s87063/repos/jit-dp-llm/llama/inference/eval_2025-10-09_19-10-15/final_test_results_seq_cls.json \
# --out /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/results/eval_results_3.json \
# --recall-target 0.84 \
# --threshold 0.4 \
# -p 0.10 0.20 \



python /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/token_length_violin.py \
--dataset /speed-scratch/a_s87063/repos/jit-dp-llm/datasets/apachejit/apachejit_total_llm_struc_meta.jsonl \
--model /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/snapshots/meta-llama/Llama-3.1-8B \
--configs /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/seq_len_confis.jsonl \
--field prompt \
--logy \
--output /speed-scratch/a_s87063/repos/jit-dp-llm/analysis/results/violin_plot

