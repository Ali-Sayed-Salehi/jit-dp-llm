jit-dp-llm is the research project for investigating how to use LLMs for just-in-time defect prediction.

# How to Setup

```bash
python -m venv venv

source venv/Scripts/activate

pip install -r requirements.txt

# If your GPU supports flash attention
pip install flash-attn --no-build-isolation
```
Create `<jit-dp-llm>/secrets/.env` and fill out the secrets.

The scripts are desgined to run completely offline. Download the LLM models you want using `<jit-dp-llm>/llama/download_model.py`.

You can download the datasets here:
https://drive.google.com/drive/folders/1QsMAn7yboltMN7gJmXDu-qKIplsjJyKq?usp=sharing

Alternatively, you can use the data_extraction directory to extract them yourself. If using Hugging Face datasets, use `<jit-dp-llm>/llama/fetch_dataset.py` to first download to your local system.

Create a `<jit-dp-llm>/llama/configs/train_config_local.yaml` from `<jit-dp-llm>/llama/configs/train_config.yaml` and change the training configs as you want.

To run the fientuning jobs either use `slurm_scripts/` or `docker/`, depending on your compute environment.