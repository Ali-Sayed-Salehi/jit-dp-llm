# ──────────────────────────────────────────────────────────────────────────────
# file: inference.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import json
import os
from utils import (
    STEmbedder, OpenAIEmbedder, JITDPResult,
    OpenAIChat, GeminiChat, HFLocalChat,
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, extract_json_maybe,
    load_repo_dotenv, build_index, retrieve_k,
    load_config_yaml, setup_logger, LOGGER_NAME,
    read_last_n_rows, sample_one_from_last_n, normalize_label,
)


def main():
    load_repo_dotenv()
    logger = setup_logger(LOGGER_NAME)

    cfg_path = os.getenv("RAG_CONFIG_PATH")  # optional override
    cfg = load_config_yaml(cfg_path)

    cmd = cfg.get("cmd", "predict")  # "build" or "predict"

    # Select embedder
    embedder_name = cfg.get("embedder", "sentence")
    embedder = (
        OpenAIEmbedder(model=cfg.get("openai_embedding_model", "text-embedding-3-small"))
        if embedder_name == "openai"
        else STEmbedder(cfg.get("st_model", "sentence-transformers/all-MiniLM-L6-v2"))
    )

    if cmd == "build":
        build_index(
            jsonl_path=cfg["build"]["data"],
            persist_dir=cfg["build"]["persist"],
            embedder=embedder,
            id_field=cfg["build"].get("id_field"),
            holdout_n=int(cfg.get("predict", {}).get("n_last", 0)),  # ensure index excludes last n
        )
        logger.info("Build completed")
        return

    # -------------------------- predict over dataset ---------------------------
    persist   = cfg["predict"]["persist"]
    data_path = cfg["predict"]["data"]                 # dataset path (JSONL)
    n_last    = int(cfg["predict"].get("n_last", 1))   # how many recent rows
    debug     = bool(cfg["predict"].get("debug", False))
    debug_seed = cfg["predict"].get("debug_seed")
    k        = int(cfg["predict"].get("k", 2))

    # Prepare target rows (dicts with at least prompt/response)
    if debug:
        target_rows = [sample_one_from_last_n(data_path, n_last, seed=debug_seed)]
    else:
        target_rows = read_last_n_rows(data_path, n_last)

    if not target_rows:
        logger.error("No targets prepared from dataset (n_last=%d).", n_last)
        raise SystemExit(1)

    # LLM backend (unchanged)
    llm_kind = cfg["predict"].get("llm", "openai")
    if llm_kind == "openai":
        llm = OpenAIChat(model=cfg["predict"].get("openai_model", "gpt-4o-mini"))
    elif llm_kind == "gemini":
        llm = GeminiChat(model=cfg["predict"].get("gemini_model", "gemini-2-flash"))
    elif llm_kind == "hf-local":
        llm = HFLocalChat(
            model_dir=cfg["predict"]["hf_model_dir"],
            torch_dtype=cfg["predict"].get("hf_dtype"),
            max_new_tokens=int(cfg["predict"].get("hf_max_new", 256)),
        )
    else:
        logger.error(f"Unknown llm kind: {llm_kind}")
        raise SystemExit(2)

    results = []
    first_logged = False

    for idx, row in enumerate(target_rows, start=1):
        target_text = row.get("prompt", "")
        gold_label = normalize_label(row.get("response", "0"))  # 0/1 -> safe/risky

        logger.info(f"[{idx}/{len(target_rows)}] Retrieving top-{k} examples…")
        shots = retrieve_k(persist, embedder, query_text=target_text, k=k, dataset_path_for_name=data_path)
        if len(shots) < 2:
            logger.error("Fewer than 2 shots retrieved — index may be empty or too small.")
            continue

        def prompt_from_doc(doc: str) -> str:
            return doc.replace("[prompt]", "").strip()

        p_a = prompt_from_doc(shots[0]["doc"]); label_a = shots[0]["label"]
        p_b = prompt_from_doc(shots[1]["doc"]); label_b = shots[1]["label"]

        prompt = USER_PROMPT_TEMPLATE.format(
            target=target_text,
            id_a=shots[0]["id"], p_a=p_a, label_a=label_a,
            id_b=shots[1]["id"], p_b=p_b, label_b=label_b,
        )

        logger.info(f"[{idx}/{len(target_rows)}] Calling LLM backend: {llm_kind}")
        raw = llm.complete(SYSTEM_PROMPT, prompt)

        # Parse STRICT JSON
        try:
            data = json.loads(raw)
            res = JITDPResult(**data)
        except Exception:
            raw2 = extract_json_maybe(raw)
            data = json.loads(raw2)
            res = JITDPResult(**data)

        res.supporting_ids = [shots[0]["id"], shots[1]["id"]]
        pred_label = res.risk

        # Print per-target JSON to stdout (optional)
        print(json.dumps(res.model_dump(), indent=2))
        results.append({
            "gold": gold_label,
            "pred": pred_label,
            "result": res.model_dump(),
        })

        # Log the FIRST target with full details to file
        if not first_logged:
            logger.info("==== FIRST TARGET DETAIL ====")
            logger.info("Target prompt:\n%s", target_text)
            logger.info("Ground truth label: %s", gold_label)
            logger.info("Predicted: %s (p=%.3f)", pred_label, res.probability)
            logger.info("Explanation: %s", res.explanation)
            logger.info("Supporting IDs: %s", res.supporting_ids)
            first_logged = True

    logger.info("Finished inference on %d targets (requested %d)", len(results), len(target_rows))


    # Optionally print an array of all results at the end:
    # print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
