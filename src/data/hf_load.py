"""HuggingFace load helpers that work on Colab (datasets 4.x + Hub auth).

Colab ships ``datasets>=4``, which dropped dataset scripts and ``trust_remote_code``.
``hotpotqa/hotpot_qa`` is parquet now. Hub catalog lookups from Colab often 404
unless ``HF_TOKEN`` is set; we then download the known parquet files or the
Wayback copy of the official JSON.
"""

from __future__ import annotations

import inspect
import json
import os
import urllib.request
from typing import Any

HOTPOT_HF_ID = "hotpotqa/hotpot_qa"
HOTPOT_HF_IDS = (HOTPOT_HF_ID, "hotpot_qa")

# Parquet layout after the 2025-08 Hub conversion (datasets>=4).
HOTPOT_PARQUET_FILES = {
    "distractor": {
        "train": (
            "distractor/train-00000-of-00002.parquet",
            "distractor/train-00001-of-00002.parquet",
        ),
        "validation": ("distractor/validation-00000-of-00001.parquet",),
    },
    "fullwiki": {
        "train": (
            "fullwiki/train-00000-of-00002.parquet",
            "fullwiki/train-00001-of-00002.parquet",
        ),
        "validation": ("fullwiki/validation-00000-of-00001.parquet",),
    },
}

# Official CMU host is down; Wayback snapshot used by vincentkoc/hotpot_qa_archive.
_WAYBACK_HOTPOT = (
    "https://web.archive.org/web/20250512032701id_/"
    "http://curtis.ml.cmu.edu/datasets/hotpot/"
)

_COLAB_HINT = (
    "On Colab this is usually missing Hub auth or datasets 4.x vs the old "
    "hotpot_qa.py script. Create a read token at https://huggingface.co/settings/tokens "
    "then, before prepare_data:\n"
    "  from google.colab import userdata\n"
    "  import os\n"
    "  os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')\n"
    "Do not pass trust_remote_code on datasets>=4 (parquet configs)."
)


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip() or None
    return None


def load_dataset_compat(path: str, *args: Any, **kwargs: Any) -> Any:
    """``load_dataset`` that injects HF_TOKEN and drops unsupported kwargs."""
    from datasets import load_dataset

    token = hf_token()
    if token and kwargs.get("token") is None:
        kwargs["token"] = token
    sig = inspect.signature(load_dataset)
    if "trust_remote_code" not in sig.parameters:
        kwargs.pop("trust_remote_code", None)
    return load_dataset(path, *args, **kwargs)


def hotpot_official_row_to_hf(row: dict[str, Any]) -> dict[str, Any]:
    """Map official HotpotQA JSON objects to the Hub parquet schema."""
    titles: list[str] = []
    sentences: list[list[str]] = []
    for item in row.get("context") or []:
        if isinstance(item, dict):
            titles.append(str(item.get("title") or ""))
            sentences.append(list(item.get("sentences") or []))
            continue
        title, sents = item
        titles.append(str(title))
        sentences.append([str(s) for s in sents])

    sf_titles: list[str] = []
    sf_ids: list[int] = []
    supporting = row.get("supporting_facts") or []
    if isinstance(supporting, dict):
        sf_titles = [str(t) for t in supporting.get("title") or []]
        sf_ids = [int(i) for i in supporting.get("sent_id") or []]
    else:
        for item in supporting:
            title, sent_id = item
            sf_titles.append(str(title))
            sf_ids.append(int(sent_id))

    return {
        "id": str(row.get("_id") or row.get("id") or ""),
        "question": row["question"],
        "answer": row["answer"],
        "type": row.get("type"),
        "level": row.get("level"),
        "supporting_facts": {"title": sf_titles, "sent_id": sf_ids},
        "context": {"title": titles, "sentences": sentences},
    }


def load_hotpot_dataset(config: str = "distractor") -> Any:
    """Load HotpotQA distractor/fullwiki with Hub, parquet, then Wayback fallbacks."""
    errors: list[str] = []

    for hf_id in HOTPOT_HF_IDS:
        try:
            print(f"Loading HotpotQA ({config}) from HuggingFace id={hf_id}...")
            return _load_hotpot_hub(hf_id, config)
        except Exception as exc:
            errors.append(f"{hf_id}: {type(exc).__name__}: {exc}")
            print(f"  Hub load failed for {hf_id}: {exc}")

    try:
        print(f"Loading HotpotQA ({config}) from Hub parquet files...")
        return _load_hotpot_parquet(config)
    except Exception as exc:
        errors.append(f"parquet: {type(exc).__name__}: {exc}")
        print(f"  Parquet load failed: {exc}")

    try:
        print(f"Loading HotpotQA ({config}) from Wayback Machine JSON...")
        return _load_hotpot_wayback(config)
    except Exception as exc:
        errors.append(f"wayback: {type(exc).__name__}: {exc}")
        print(f"  Wayback load failed: {exc}")

    detail = " | ".join(errors)
    raise RuntimeError(
        f"Could not load HotpotQA config={config!r}. {detail}\n{_COLAB_HINT}"
    )


def _load_hotpot_hub(hf_id: str, config: str) -> Any:
    kwargs: dict[str, Any] = {}
    # datasets 2.x still needs this for the old hotpot_qa.py script.
    kwargs["trust_remote_code"] = True
    try:
        return load_dataset_compat(hf_id, config, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        return load_dataset_compat(hf_id, config, **kwargs)


def _load_hotpot_parquet(config: str) -> Any:
    from huggingface_hub import hf_hub_download

    files = HOTPOT_PARQUET_FILES.get(config)
    if not files:
        raise ValueError(f"Unknown Hotpot config {config!r}; expected distractor or fullwiki")
    token = hf_token()
    data_files: dict[str, list[str]] = {}
    for split, names in files.items():
        data_files[split] = [
            hf_hub_download(
                repo_id=HOTPOT_HF_ID,
                filename=name,
                repo_type="dataset",
                token=token,
            )
            for name in names
        ]
    return load_dataset_compat("parquet", data_files=data_files)


def _load_hotpot_wayback(config: str) -> Any:
    from datasets import Dataset, DatasetDict

    urls = {
        "train": _WAYBACK_HOTPOT + "hotpot_train_v1.1.json",
        "validation": _WAYBACK_HOTPOT + f"hotpot_dev_{config}_v1.json",
    }
    splits: dict[str, Any] = {}
    for split, url in urls.items():
        print(f"  fetching {url} ...")
        with urllib.request.urlopen(url, timeout=300) as resp:
            raw = json.load(resp)
        if not isinstance(raw, list):
            raise ValueError(f"Unexpected Hotpot JSON at {url}")
        splits[split] = Dataset.from_list([hotpot_official_row_to_hf(row) for row in raw])
    return DatasetDict(splits)
