jit-dp-llm is the research project for investigating how to use LLMs, specifically Llama, for just-in-time defect prediction.

# How to Setup

```bash
python -m venv venv

source venv/Scripts/activate

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
Create `<jit-dp-llm>/secrets/.env` and fill out the secrets.

Contact the authors to provide the datsets for you, alternatively, you can use the data_extraction directory to create them yourself

Create a `train_config_local.yaml` from `<jit-dp-llm>/llama/configs/train_config.yaml` and change the configs as you want.

To run the fientuning jobs either use `slurm_scripts/` or `docker/`, depending on your compute environment.