# bugbug glue: training data in, risk scores out

Scripts that connect this repo to bugbug's performance alerts data pipeline
without changing bugbug or Firefox. The pipeline (bugbug repo,
`docs/perf-alerts.md`) labels which autoland commits caused confirmed
performance regressions; its `export` command writes those commits with their
patches. The scripts here turn that into a training set, prepare commits for
GPU scoring on the cluster, and turn the scores back into the pipeline's
`perf_commit_risk.json` records, so bugbug's `calibrate` command and the perf
scheduling simulator can judge a new model exactly as they judge the deployed
one.

Two facts shape the design. bugbug's model class exposes the prompt builder
the prediction endpoint uses (cleaned commit subject plus the structured diff
between the eight `<FILE>`/`<ADDED>`/... tags), so both the training rows and
the scoring prompts are built with it and the model never sees a different
text in production. And bugbug's loader pins inference to the CPU and reads
patches from a Mercurial clone, so GPU scoring happens here: prompts are built
locally next to the clone, scored on the cluster, and merged back locally.

## Scripts

| Script | Runs where | Does |
| --- | --- | --- |
| `build_training_set.py` | locally, bugbug virtualenv | bugbug export -> `<commit_id, prompt, response>` JSON lines in push-date order |
| `build_scoring_set.py` | locally, bugbug virtualenv + hg clone | commits of a period -> push fields, bugbug's ignore reason, prompt |
| `../../llama/score_commits.py` | cluster GPU | prompts -> `prob`, `class`, `risk_score` per commit, resumable |
| `to_commit_risk.py` | locally | scores -> `perf_commit_risk.json` records under a model version, merged into the archive |
| `../../llama/export_bugbug_checkpoint.py` | either | trained checkpoint -> `perfregressionpredictormodel/` with bugbug's label names and metadata, plus the `.tar.zst` |

The local scripts import bugbug, so run them with the bugbug virtualenv's
interpreter (`~/repos/bugbug/.venv/bin/python`). The cluster scripts need only
torch, transformers and peft.

## Windows

The datasets reach from October 2024 to October 2026. The split keeps every
evaluation out of the model's sample:

| Purpose | Window | Why |
| --- | --- | --- |
| training (`export --until 2026-01-01`) | Oct 2024 to Dec 2025 | the loader's `train_fraction`/`eval_fraction` cut it 88/12 chronologically; the eval slice selects checkpoints |
| model test, calibration fit, optimizer tuning | Jan to May 2026 | scored by the new model; recall and precision in the top 1 and 2 percent at the real base rate decide between models; bugbug fits the calibration here |
| simulation eval | Jun to Oct 2026 | same window as the existing studies, where the recorded production schedule exists |

Within training, `recall@top_5%` on the eval slice selects checkpoints; the
top 1 and 2 percent are reported but rest on a dozen commits there. The
training rows have five negatives per culprit (the export's default), so
their recall-at-top-k is not comparable with the 1.4 percent base rate of the
real stream; only the January to May scores are.

## Steps

1. Export the training window from the pipeline's data directory, then build
   the training set:

   ```sh
   cd ~/perf-alerts-full
   bugbug-data-perf-alerts export --settle-days 7 --until 2026-01-01 --test-fraction 0 --repo-dir ~/repos/mozilla-unified
   cd ~/repos/jit-dp-llm
   ~/repos/bugbug/.venv/bin/python data_extraction/bugbug/build_training_set.py \
       --export ~/perf-alerts-full/data/perf_regression_commits.json \
       --output datasets/mozilla_perf_v2/perf_bugbug_struc.jsonl
   ```

2. Train on the cluster with one of the `*_perf_v2.yaml` templates
   (`slurm_scripts/speed/train_perf_v2.sh <config>`). For a LoRA run (Qwen),
   merge the adapter with `llama/merge_lora.py --task seq-cls` afterwards.

3. Prepare the commits to score, one file per model context length (the
   patch is capped at 64 characters per token before structuring, as bugbug
   does), and copy it to the cluster:

   ```sh
   ~/repos/bugbug/.venv/bin/python data_extraction/bugbug/build_scoring_set.py \
       --data-dir ~/perf-alerts-full/data --repo-dir ~/repos/mozilla-unified \
       --since 2026-01-01 --until 2026-10-28 --max-length 512 8192 16384 \
       --output "datasets/mozilla_perf_v2/scoring_2026_{max_length}.jsonl"
   ```

4. Score on the cluster: `slurm_scripts/speed/score_commits.sh <checkpoint>
   <scoring set> <scores> [batch size] [base model]`. Rerun to resume.

5. Package the checkpoint and merge the scores back, then let the pipeline
   calibrate and the simulator judge:

   ```sh
   python llama/export_bugbug_checkpoint.py --model_path <run>/model \
       --output_dir LLMs/trained/sequence-classification/perf_mbert_v2 \
       --model_version perf-regression-predictor-v2-mbert-2026-11 \
       --dataset datasets/mozilla_perf_v2/perf_bugbug_struc.jsonl \
       --training_config llama/configs/templates/train_config_speed_mbert_perf_v2.yaml
   python data_extraction/bugbug/to_commit_risk.py \
       --scoring-set datasets/mozilla_perf_v2/scoring_2026_8k.jsonl \
       --scores datasets/mozilla_perf_v2/scores_2026_mbert.jsonl \
       --model-dir LLMs/trained/sequence-classification/perf_mbert_v2/perfregressionpredictormodel \
       --archive ~/perf-alerts-full/data/perf_commit_risk.json
   cd ~/perf-alerts-full
   bugbug-data-perf-alerts calibrate --settle-days 7
   bugbug-perf-scheduling-simulator optimize --baseline rule --since 2026-01-01 \
       --optimize-until 2026-06-01 --model-version perf-regression-predictor-v2-mbert-2026-11 \
       --bootstrap 1000 --output results/optimize_v2_mbert.json --markdown results/optimize_v2_mbert.md
   ```

   The calibration record for the new version reports precision and recall in
   the top 1, 2 and 5 percent of its holdout; the optimizer reports the
   batching trade-off against production.

Deploying a model means uploading the `.tar.zst` to a new versioned path in
the models bucket and pointing bugbug's `artifact_url` at it, a one-line
change outside this repo. Only models that score a commit on a CPU within a
decision task's budget are deployable; a 1.5B decoder at long context is
borderline, the encoders are fine.
