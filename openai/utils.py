
# ──────────────────────────────────────────────────────────────────────────────
# file: utils.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import json
import os
import logging
from datetime import datetime
from typing import List, Any, Dict
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from pathlib import Path
from collections import deque
import random
import re
from pathlib import Path

LOGGER_NAME = "inference"
logger = logging.getLogger(f"{LOGGER_NAME}.utils")

# Optional deps (import lazily/defensively)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*a, **k):  # type: ignore
        return None

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

import chromadb
from chromadb.config import Settings

# Embedding backends
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    from openai import OpenAI as OpenAIClient  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


# ========== Paths & .env =====================================================
_ENV_LOADED = False

def repo_openai_dir() -> Path:
    """Absolute path to repo_root/openai (this file lives here)."""
    return Path(__file__).resolve().parent

def load_repo_dotenv() -> None:
    """Load .env from repo_root/secrets/.env relative to this file."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = repo_openai_dir().parents[0] / "secrets" / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    _ENV_LOADED = True


# ========== YAML config loader ===============================================
DEFAULT_CONFIG_PATH = repo_openai_dir() / "config" / "inference_config.yaml"

def load_config_yaml(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load YAML config dict from openai/config/inference_config.yaml by default.
    Raises RuntimeError if PyYAML missing or file not found/parsable.
    """
    if not _HAS_YAML:
        raise RuntimeError("pyyaml not installed: pip install pyyaml")
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise RuntimeError(f"Config YAML not found at: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError("Config YAML must deserialize to a mapping (dict)")
    return data


# ========== Logger setup =====================================================
def setup_logger(name: str = "rag") -> logging.Logger:
    log_dir = repo_openai_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # make idempotent: only add handlers if none exist
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.propagate = False

        logger.info(f"Logging to {log_file}")
    return logger


# ========== LLM backends =====================================================
class LLM(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...

class OpenAIChat(LLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        if not _HAS_OPENAI:
            raise RuntimeError("openai not installed: pip install openai")
        load_repo_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY missing in env or .env")
        self.client = OpenAIClient()
        self.model = model
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

class GeminiChat(LLM):
    def __init__(self, model: str = "gemini-2-flash"):
        import google.generativeai as genai  # type: ignore
        load_repo_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing in env or .env")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        resp = self.model.generate_content(prompt)
        return getattr(resp, "text", "") or ""

class HFLocalChat(LLM):
    """Local inference using a Hugging Face snapshot on disk (via transformers.pipeline)."""
    def __init__(
        self,
        model_dir: str,
        torch_dtype: str | None = None,
        max_new_tokens: int = 256,
    ):
        import torch  # type: ignore
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        model_kwargs = dict(
            device_map="auto",
            torch_dtype=getattr(torch, torch_dtype) if torch_dtype else None,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
        self.pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer)
        self.max_new_tokens = max_new_tokens
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_text = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n\nASSISTANT:"

        outputs = self.pipe(
            prompt_text,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
        )
        return (outputs[0]["generated_text"] or "").strip()


# ========== Embedders ========================================================
class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]: ...

class STEmbedder(Embedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers not installed: pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        if not _HAS_OPENAI:
            raise RuntimeError("openai not installed: pip install openai")
        load_repo_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY missing in env or .env")
        self.client = OpenAIClient()
        self.model = model
    def embed(self, texts: List[str]) -> List[List[float]]:
        out = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in out.data]


# ========== Schema & prompts =================================================
class JITDPResult(BaseModel):
    risk: str = Field(description="'risky' or 'safe'")
    probability: float = Field(ge=0, le=1)
    explanation: str
    supporting_ids: list[str]

def normalize_label(v: Any) -> str:
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "risky", "bug", "defect"}:
        return "risky"
    return "safe"

SYSTEM_PROMPT = (
    "You are a senior software engineering risk analyst. "
    "Given a software change, decide if it is RISKY or SAFE and explain briefly. "
    "Use the provided past examples as guidance. Output STRICT JSON: "
    "{'risk': 'risky'|'safe', 'probability': 0..1, 'explanation': str, 'supporting_ids': [str, str]}"
)

USER_PROMPT_TEMPLATE = """
You will be given:
1) The target commit which includes commits message and code diff.
2) Two similar past examples with their labels.

Target commit:
---
{target}
---

Similar examples:
[Example A] id={id_a}  label={label_a}
commit:
{p_a}

[Example B] id={id_b}  label={label_b}
commit:
{p_b}

Task:
- Predict whether the target commit is risky or safe.
- Give a short explanation (2–5 sentences) that references patterns from the examples when relevant.
- Output STRICT JSON only (no markdown, no prose) with:
  {{"risk": "risky"|"safe", "probability": float 0..1, "explanation": "...", "supporting_ids": ["{id_a}", "{id_b}"]}}
"""

def extract_json_maybe(s: str) -> str:
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return s


# ========== Vector DB operations ============================================
COLLECTION_NAME_BASE = "jitdp"

def dataset_id_from_path(dataset_path: str) -> str:
    stem = Path(dataset_path).stem                      # e.g., jit_defects4j_small_llm_struc
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", stem)      # sanitize

def make_collection_name(dataset_path: str) -> str:
    return f"{COLLECTION_NAME_BASE}_{dataset_id_from_path(dataset_path)}"

def get_collection(persist_dir: str, name: str):
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(name)


def _flush(col, ids, texts, metas, embedder: Embedder):
    vectors = embedder.embed(texts)
    col.upsert(ids=ids, embeddings=vectors, metadatas=metas, documents=texts)

def _jsonl_row_to_text(row: Dict[str, Any]) -> str:
    prompt = (row.get("prompt") or "").strip()
    return f"[prompt] \n{prompt}"

def build_index(
    jsonl_path: str,
    persist_dir: str,
    embedder: Embedder,
    id_field: str | None = None,
    batch_size: int = 256,
    holdout_n: int = 0,   # NEW: do not index the last holdout_n rows
):
    os.makedirs(persist_dir, exist_ok=True)
    collection_name = make_collection_name(jsonl_path)
    col = get_collection(persist_dir, collection_name)

    # Pass 1: count total rows
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _ in f:
            total += 1

    cutoff = max(0, total - max(0, int(holdout_n)))  # index rows [0 .. cutoff-1]
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Indexing %s rows out of %s (holdout_n=%s, cutoff=%s)",
                cutoff, total, holdout_n, cutoff)

    # Pass 2: index only up to cutoff-1
    buffer_ids: List[str] = []
    buffer_texts: List[str] = []
    buffer_metas: List[Dict[str, Any]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= cutoff:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get(id_field)) if id_field and row.get(id_field) is not None else str(idx)
            text = _jsonl_row_to_text(row)
            label = normalize_label(row.get("response", "0"))
            meta = {"id": rid, "label": label, "prompt_excerpt": (row.get("prompt", "") or "")[:200]}
            buffer_ids.append(rid); buffer_texts.append(text); buffer_metas.append(meta)
            if len(buffer_ids) >= batch_size:
                _flush(col, buffer_ids, buffer_texts, buffer_metas, embedder)
                buffer_ids, buffer_texts, buffer_metas = [], [], []

    if buffer_ids:
        _flush(col, buffer_ids, buffer_texts, buffer_metas, embedder)

    logger.info("Indexed %s -> %s (collection '%s')", jsonl_path, persist_dir, COLLECTION_NAME)


def retrieve_k(
    persist_dir: str,
    embedder: Embedder,
    query_text: str,
    k: int = 2,
    collection_name: str | None = None,
    dataset_path_for_name: str | None = None,
):
    if collection_name is None:
        if not dataset_path_for_name:
            raise ValueError("Provide collection_name or dataset_path_for_name to resolve the collection.")
        collection_name = make_collection_name(dataset_path_for_name)

    col = get_collection(persist_dir, collection_name)
    qvec = embedder.embed([query_text])[0]
    res = col.query(query_embeddings=[qvec], n_results=k)
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "doc": res["documents"][0][i],
            "label": res["metadatas"][0][i].get("label", "safe"),
        })
    return out


def _jsonl_iter(path: str):
    """Yield parsed JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def read_last_n_prompts(jsonl_path: str, n: int) -> list[str]:
    """
    Return the last n 'prompt' strings from the dataset.
    Uses a deque for memory efficiency.
    """
    dq = deque(maxlen=max(1, n))
    for row in _jsonl_iter(jsonl_path):
        dq.append(row)
    targets = [r.get("prompt", "") for r in dq]
    logger.info("Prepared %d targets from last %d rows", len(targets), n)
    return targets

def sample_debug_prompt_from_tail(jsonl_path: str, tail_ratio: float = 0.10, seed: int | None = None) -> str:
    """
    Pick 1 random 'prompt' from the last `tail_ratio` portion of rows (default 10%).
    Uses a two-pass approach to avoid loading entire file into memory.
    """
    # Pass 1: count
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for _ in f:
            total += 1
    if total == 0:
        raise RuntimeError(f"No rows found in dataset: {jsonl_path}")
    start_idx = max(0, int(total * (1.0 - tail_ratio)))

    # Random choice in [start_idx, total-1]
    rng = random.Random(seed)
    pick_idx = rng.randint(start_idx, total - 1)

    # Pass 2: fetch that row
    cur = -1
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            cur += 1
            if cur == pick_idx:
                row = json.loads(line)
                prompt = row.get("prompt", "")
                logger.info(
                    "Debug mode: picked row %d from tail [%d..%d) of %d",
                    pick_idx, start_idx, total, total
                )
                return prompt
    # Fallback (shouldn't hit)
    raise RuntimeError("Failed to pick a debug row")


def read_last_n_rows(jsonl_path: str, n: int) -> list[dict]:
    """
    Return the last n rows (full JSON objects) from a JSONL file.
    """
    n = max(1, int(n))
    dq: deque = deque(maxlen=n)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dq.append(json.loads(line))
    return list(dq)

def sample_one_from_last_n(jsonl_path: str, n: int, seed: int | None = None) -> dict:
    """
    Pick 1 random row from the last n rows of the dataset.
    """
    tail = read_last_n_rows(jsonl_path, n)
    if not tail:
        raise RuntimeError(f"No rows available to sample in last {n} rows of {jsonl_path}")
    rng = random.Random(seed)
    return rng.choice(tail)
