#!/encs/bin/tcsh

#SBATCH --job-name=download-model-speed
#SBATCH --output=/speed-scratch/a_s87063/repos/perf-pilot/slurm_jobs/%x-%j.out
#SBATCH --partition=ps,pg,pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu20

echo "Preparing training environment"
source /speed-scratch/a_s87063/repos/perf-pilot/slurm_scripts/speed/train_prepare.csh

echo "Running the download/transform script ..."
# python /speed-scratch/a_s87063/repos/perf-pilot/llama/download_model.py --model_id meta-llama/Llama-4-Scout-17B-16E --model_head causal-lm
# python /speed-scratch/a_s87063/repos/perf-pilot/llama/download_model.py --model_id syzymon/long_llama_code_7b --model_head sequence-classification
# python /speed-scratch/a_s87063/repos/perf-pilot/llama/attach_classification_head.py --base_lm_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/causal-lm/meta-llama/Llama-3.1-8B --save_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification/custom
python /speed-scratch/a_s87063/repos/perf-pilot/llama/attach_classification_head_llama4.py --base_lm_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/causal-lm/meta-llama/Llama-4-Scout-17B-16E --save_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification
accelerate launch --config_file=/speed-scratch/a_s87063/repos/perf-pilot/llama/configs/accelerate_config_deepspeed.yaml --num_machines=$SLURM_NNODES --machine_rank=$SLURM_NODEID /speed-scratch/a_s87063/repos/perf-pilot/llama/attach_classification_head_llama4.py --base_lm_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/causal-lm/meta-llama/Llama-4-Scout-17B-16E --save_path /speed-scratch/a_s87063/repos/perf-pilot/LLMs/pretrained/sequence-classification
echo "download/transform finished"
