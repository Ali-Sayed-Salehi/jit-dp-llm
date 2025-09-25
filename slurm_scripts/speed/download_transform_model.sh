#!/encs/bin/tcsh

#SBATCH --job-name=download-model-speed
#SBATCH --output=/speed-scratch/a_s87063/repos/jit-dp-llm/slurm_jobs/%x-%j.out
#SBATCH --partition=ps,pg,pt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=pcr
#SBATCH --constraint=gpu20

echo "Preparing training environment"
source /speed-scratch/a_s87063/repos/jit-dp-llm/slurm_scripts/speed/train_prepare.csh

setenv HF_HUB_ENABLE_HF_TRANSFER 1

echo "Running the download/transform script ..."
# python /speed-scratch/a_s87063/repos/jit-dp-llm/llama/download_model.py --model_id meta-llama/Llama-3.1-8B --model_head sequence-classification --save_path /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/pretrained/sequence-classification/custom/bf32
# python /speed-scratch/a_s87063/repos/jit-dp-llm/llama/download_model.py --model_id syzymon/long_llama_code_7b --model_head sequence-classification
# python /speed-scratch/a_s87063/repos/jit-dp-llm/llama/attach_classification_head.py --base_lm_path /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/pretrained/causal-lm/meta-llama/Llama-3.1-8B --save_path /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/pretrained/sequence-classification/custom
# python /speed-scratch/a_s87063/repos/jit-dp-llm/llama/attach_classification_head.py --llama4 --base_lm_path /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/pretrained/causal-lm/meta-llama/Llama-4-Scout-17B-16E --save_path /speed-scratch/a_s87063/repos/jit-dp-llm/LLMs/pretrained/sequence-classification/custom
python /speed-scratch/a_s87063/repos/jit-dp-llm/llama/download_model.py --model_id bigcode/starcoder2-7b
echo "download/transform finished"
