"""Real Wikipedia evidence for the single-hop split (DPR / KILT family).

Primary source is ``Tevatron/wikipedia-nq``: DPR 100-word Wikipedia passages
with NQ questions and answers (Karpukhin et al. 2020). That is the reviewer-
expected NQ evidence format without downloading the 21M ``wiki_dpr`` dump.

Fallbacks (still real passages, never answer-anchors):
  1. ``Tevatron/wikipedia-trivia``
  2. ``Tevatron/wikipedia-squad``
  3. ``rajpurkar/squad`` article contexts
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Iterable, Iterator


from .loaders import NQ_ANCHOR_SOURCE, NQ_WIKI_NEG_SOURCE, NQ_WIKI_SOURCE


def _stable_id(*parts: str) -> str:
    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return h

SINGLE_HOP_DATASETS = ("natural_questions", "trivia_qa", "squad")

# Colab-safe: per-query DPR negatives only. The 80k Hotpot pool still fills the index.
DEFAULT_NEGATIVES_PER_QUERY = 8

# SQuAD is stored article-grouped. A 150-question prefix collapsed to 16 golds.
# Refuse a single-hop eval thinner than ~30 distinct gold articles per 150 questions.
MIN_DISTINCT_GOLD_ARTICLES_PER_150 = 30


def min_distinct_gold_articles(n_questions: int) -> int:
    if int(n_questions) <= 0:
        return 0
    return max(1, int(round(MIN_DISTINCT_GOLD_ARTICLES_PER_150 * int(n_questions) / 150)))


def distinct_single_hop_gold_articles(examples: Iterable[dict[str, Any]]) -> int:
    titles: set[str] = set()
    for example in examples:
        if str(example.get("dataset") or "") not in SINGLE_HOP_DATASETS:
            continue
        for title in example.get("supporting_titles") or []:
            t = " ".join(str(title).split())
            if t:
                titles.add(t.casefold())
    return len(titles)


def round_robin_by_title(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Seeded shuffle within and across titles, then round-robin.

    SQuAD JSONL is article-grouped, so taking the first N questions is a
    single-topic slice. This order is what ``examples_from_rows`` consumes.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        title = str(row.get("title") or "").strip() or "_untitled"
        buckets[title].append(row)
    rng = random.Random(int(seed))
    titles = list(buckets)
    rng.shuffle(titles)
    for title in titles:
        rng.shuffle(buckets[title])
    out: list[dict[str, Any]] = []
    remaining = True
    depth = 0
    while remaining:
        remaining = False
        for title in titles:
            bucket = buckets[title]
            if depth < len(bucket):
                out.append(bucket[depth])
                remaining = True
        depth += 1
    return out


def looks_like_answer_anchor(question: str, answers: Iterable[str] | None, text: str) -> bool:
    """True for the old leak: question text plus ``The answer is {gold}``."""
    q = " ".join((question or "").split()).lower()
    t = " ".join((text or "").split()).lower()
    if not q or not t:
        return False
    if t.startswith(q) and "the answer is" in t[: max(len(q) + 120, 240)]:
        return True
    for ans in answers or []:
        a = " ".join(str(ans).split()).lower()
        if a and t == f"{q} the answer is {a}. according to reference sources, the answer is {a}.":
            return True
    return False


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict) and "text" in value:
        return as_str_list(value.get("text"))
    out: list[str] = []
    try:
        items = list(value)
    except TypeError:
        return out
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            out.extend(as_str_list(item.get("text") or item.get("answer") or item.get("value")))
    return out


def as_ctx_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        texts = value.get("text")
        if isinstance(texts, list):
            n = len(texts)
            rows: list[dict[str, Any]] = []
            for i in range(n):
                rows.append(
                    {
                        key: (val[i] if isinstance(val, list) and i < len(val) else val)
                        for key, val in value.items()
                    }
                )
            return rows
        if value.get("text") or value.get("docid") or value.get("title"):
            return [value]
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, dict)]
    return []


def dpr_ctx_to_passage(ctx: dict[str, Any], *, gold: bool) -> dict[str, Any] | None:
    text = " ".join(str(ctx.get("text") or "").split()).strip()
    if not text:
        return None
    docid = str(ctx.get("docid") or ctx.get("passage_id") or "").strip()
    title = str(ctx.get("title") or "").strip()
    if not docid:
        docid = _stable_id("dpr", title, text[:80])
    if not title:
        title = docid
    return {
        "passage_id": f"dpr_{docid}",
        "title": title,
        "text": text,
        "source": NQ_WIKI_SOURCE if gold else NQ_WIKI_NEG_SOURCE,
        "is_gold_support": gold,
    }


def _row_question(row: dict[str, Any]) -> str:
    return str(row.get("query") or row.get("question") or row.get("input") or "").strip()


def tevatron_row_to_example(
    row: dict[str, Any],
    *,
    split: str,
    dataset: str,
    index: int,
    negatives_per_query: int = DEFAULT_NEGATIVES_PER_QUERY,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Convert one Tevatron DPR row to an example plus Wikipedia passages.

    Returns None if there is no usable gold passage (including leaked anchors).
    """
    question = _row_question(row)
    answers = as_str_list(row.get("answers") or row.get("answer"))
    if not question or not answers:
        return None

    golds: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ctx in as_ctx_list(row.get("positive_passages") or row.get("positive_ctxs")):
        passage = dpr_ctx_to_passage(ctx, gold=True)
        if passage is None:
            continue
        if looks_like_answer_anchor(question, answers, passage["text"]):
            continue
        if passage["passage_id"] in seen:
            continue
        seen.add(passage["passage_id"])
        golds.append(passage)
    if not golds:
        return None

    passages = list(golds)
    n_neg = max(0, int(negatives_per_query))
    for ctx in as_ctx_list(row.get("negative_passages") or row.get("hard_negative_ctxs") or row.get("negative_ctxs")):
        if n_neg <= 0:
            break
        passage = dpr_ctx_to_passage(ctx, gold=False)
        if passage is None or passage["passage_id"] in seen:
            continue
        if looks_like_answer_anchor(question, answers, passage["text"]):
            continue
        seen.add(passage["passage_id"])
        passages.append(passage)
        n_neg -= 1

    qid = str(row.get("query_id") or row.get("id") or index)
    example = {
        "id": f"{dataset}_{split}_{qid}",
        "dataset": dataset,
        "split": split,
        "question": question,
        "answers": answers,
        "answer": answers[0],
        "type": "single_hop",
        "level": None,
        "supporting_titles": [p["title"] for p in golds],
        "local_passage_ids": [p["passage_id"] for p in golds],
    }
    return example, passages


def squad_row_to_example(
    row: dict[str, Any],
    *,
    split: str,
    index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    question = _row_question(row)
    answers = as_str_list(row.get("answers") or row.get("answer"))
    context = " ".join(str(row.get("context") or "").split()).strip()
    title = str(row.get("title") or "").strip() or "squad"
    if not question or not answers or not context:
        return None
    if looks_like_answer_anchor(question, answers, context):
        return None
    passage = {
        "passage_id": f"squad_{_stable_id('squad', title, context[:80])}",
        "title": title,
        "text": context,
        "source": NQ_WIKI_SOURCE,
        "is_gold_support": True,
    }
    qid = str(row.get("id") or index)
    example = {
        "id": f"squad_{split}_{qid}",
        "dataset": "squad",
        "split": split,
        "question": question,
        "answers": answers,
        "answer": answers[0],
        "type": "single_hop",
        "level": None,
        "supporting_titles": [title],
        "local_passage_ids": [passage["passage_id"]],
    }
    return example, [passage]


def examples_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    split: str,
    n: int,
    kind: str,
    dataset: str,
    negatives_per_query: int = DEFAULT_NEGATIVES_PER_QUERY,
    max_scan: int | None = None,
    seed: int = 42,
    diversify_by_title: bool | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Take ``n`` examples that have real gold passages. Skip leaky / empty rows.

    SQuAD is article-grouped. Unless ``diversify_by_title`` is False, squad
    rows are materialized and round-robined by title before taking N.
    """
    want = int(n)
    if diversify_by_title is None:
        diversify_by_title = kind == "squad"
    if diversify_by_title:
        materialized = [dict(raw) for raw in rows]
        rows = round_robin_by_title(materialized, seed)
        scan_cap = len(rows)
    else:
        scan_cap = int(max_scan) if max_scan is not None else max(want * 25, want + 50)
    examples: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    scanned = 0
    skipped = 0

    for raw in rows:
        scanned += 1
        row = dict(raw)
        converted = (
            squad_row_to_example(row, split=split, index=scanned - 1)
            if kind == "squad"
            else tevatron_row_to_example(
                row,
                split=split,
                dataset=dataset,
                index=scanned - 1,
                negatives_per_query=negatives_per_query,
            )
        )
        if converted is None:
            skipped += 1
        else:
            example, passages = converted
            examples.append(example)
            for passage in passages:
                pid = passage["passage_id"]
                existing = by_id.get(pid)
                if existing is None or (passage.get("is_gold_support") and not existing.get("is_gold_support")):
                    by_id[pid] = passage
        if len(examples) >= want or scanned >= scan_cap:
            break

    kept = examples[:want]
    stats = {
        "n_requested": want,
        "n_kept": len(kept),
        "n_scanned": scanned,
        "n_skipped": skipped,
        "n_passages": len(by_id),
        "n_gold_articles": distinct_single_hop_gold_articles(kept),
        "hit_quota": len(kept) >= want,
        "diversify_by_title": bool(diversify_by_title),
    }
    if len(kept) < want:
        raise RuntimeError(
            f"Need {want} {dataset} {split} examples with real Wikipedia passages; "
            f"kept {len(kept)} after scanning {scanned} rows ({skipped} skipped)."
        )
    return kept, list(by_id.values()), stats


def merge_passages(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedup by passage_id. A gold copy always wins over a distractor copy."""
    by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        for passage in group:
            pid = passage["passage_id"]
            existing = by_id.get(pid)
            if existing is None or (passage.get("is_gold_support") and not existing.get("is_gold_support")):
                by_id[pid] = passage
    return list(by_id.values())


def count_leaky_anchors(passages: Iterable[dict[str, Any]]) -> int:
    n = 0
    for passage in passages:
        source = str(passage.get("source") or "")
        title = str(passage.get("title") or "")
        if source == NQ_ANCHOR_SOURCE or title.startswith("NQ anchor"):
            n += 1
    return n


_TEVATRON_CHAIN: tuple[dict[str, Any], ...] = (
    {
        "dataset": "natural_questions",
        "hf_id": "Tevatron/wikipedia-nq",
        "kind": "tevatron",
        "train_split": "train",
        "eval_splits": ("test", "dev", "validation"),
        "label": "DPR Wikipedia 100-word passages (Tevatron/wikipedia-nq)",
    },
    {
        "dataset": "trivia_qa",
        "hf_id": "Tevatron/wikipedia-trivia",
        "kind": "tevatron",
        "train_split": "train",
        "eval_splits": ("dev", "validation", "test"),
        "label": "TriviaQA Wikipedia passages (Tevatron/wikipedia-trivia)",
    },
    {
        "dataset": "squad",
        "hf_id": "Tevatron/wikipedia-squad",
        "kind": "tevatron",
        "train_split": "train",
        "eval_splits": ("dev", "validation", "test"),
        "label": "SQuAD-open Wikipedia passages (Tevatron/wikipedia-squad)",
    },
    {
        "dataset": "squad",
        "hf_id": "rajpurkar/squad",
        "kind": "squad",
        "train_split": "train",
        "eval_splits": ("validation",),
        "label": "SQuAD article contexts (rajpurkar/squad)",
    },
)


def _iter_hf_split(hf_id: str, split: str, streaming: bool) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    # Tevatron sources are dataset *scripts*. datasets 2.x needs this flag;
    # datasets 3.x removed script support entirely (pin datasets<3.0).
    # HF_TOKEN is for Hub rate limits, not this loader failure mode.
    kwargs = {"split": split, "streaming": streaming, "trust_remote_code": True}
    try:
        ds = load_dataset(hf_id, **kwargs)
    except Exception:
        if not streaming:
            raise
        kwargs["streaming"] = False
        ds = load_dataset(hf_id, **kwargs)
    for row in ds:
        yield dict(row)


def _take_hf_source(
    spec: dict[str, Any],
    split_role: str,
    n: int,
    negatives_per_query: int,
    streaming: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split_names = (spec["train_split"],) if split_role == "train" else tuple(spec["eval_splits"])
    # SQuAD is small and article-grouped; materialize so we can round-robin titles.
    use_stream = bool(streaming) and spec["kind"] != "squad"
    errors: list[str] = []
    for split in split_names:
        try:
            print(f"  loading {spec['hf_id']} split={split} ({split_role}, n={n})...")
            examples, passages, stats = examples_from_rows(
                _iter_hf_split(spec["hf_id"], split, use_stream),
                split="train" if split_role == "train" else "eval",
                n=n,
                kind=spec["kind"],
                dataset=spec["dataset"],
                negatives_per_query=negatives_per_query,
                seed=seed,
            )
            stats.update(
                {
                    "hf_id": spec["hf_id"],
                    "hf_split": split,
                    "dataset": spec["dataset"],
                    "label": spec["label"],
                    "kind": spec["kind"],
                }
            )
            return examples, passages, stats
        except Exception as exc:
            errors.append(f"{spec['hf_id']}:{split}: {exc}")
            continue
    raise RuntimeError(" / ".join(errors) if errors else f"no splits for {spec['hf_id']}")


def load_single_hop_with_real_passages(
    n_train: int,
    n_eval: int,
    *,
    negatives_per_query: int = DEFAULT_NEGATIVES_PER_QUERY,
    streaming: bool = True,
    preferred_hf_id: str | None = None,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load train+eval single-hop items with Wikipedia evidence.

    Tries DPR NQ first, then TriviaQA / SQuAD. Never plants answer-anchor passages.
    """
    chain = list(_TEVATRON_CHAIN)
    if preferred_hf_id:
        preferred = [s for s in chain if s["hf_id"] == preferred_hf_id]
        rest = [s for s in chain if s["hf_id"] != preferred_hf_id]
        chain = preferred + rest

    errors: list[str] = []
    need_eval_articles = min_distinct_gold_articles(n_eval)
    for spec in chain:
        try:
            print(f"Trying single-hop source: {spec['label']}")
            train_ex, train_psg, train_stats = _take_hf_source(
                spec, "train", n_train, negatives_per_query, streaming, seed
            )
            eval_ex, eval_psg, eval_stats = _take_hf_source(
                spec, "eval", n_eval, negatives_per_query, streaming, seed
            )
            passages = merge_passages(train_psg, eval_psg)
            if count_leaky_anchors(passages):
                raise RuntimeError("refusing answer-anchor passages in the single-hop corpus")
            n_eval_articles = distinct_single_hop_gold_articles(eval_ex)
            if n_eval and n_eval_articles < need_eval_articles:
                raise RuntimeError(
                    f"{spec['hf_id']} eval has {n_eval_articles} distinct gold articles; "
                    f"need >= {need_eval_articles} for {n_eval} questions "
                    f"(~{MIN_DISTINCT_GOLD_ARTICLES_PER_150} per 150). "
                    "Refusing a single-topic single-hop slice."
                )
            meta = {
                "dataset": spec["dataset"],
                "hf_id": spec["hf_id"],
                "label": spec["label"],
                "kind": spec["kind"],
                "nq_corpus": "dpr_wikipedia_w100" if spec["dataset"] == "natural_questions" else spec["dataset"],
                "train": train_stats,
                "eval": eval_stats,
                "n_passages": len(passages),
                "n_gold": sum(1 for p in passages if p.get("is_gold_support")),
                "n_eval_gold_articles": n_eval_articles,
            }
            print(
                f"Single-hop source OK: {spec['label']} "
                f"(train={len(train_ex)} eval={len(eval_ex)} wiki_passages={len(passages)} "
                f"eval_gold_articles={n_eval_articles})"
            )
            return train_ex, eval_ex, passages, meta
        except Exception as exc:
            msg = f"{spec['hf_id']}: {exc}"
            print(f"  failed ({msg})")
            errors.append(msg)
            continue
    raise RuntimeError(
        "Could not load a single-hop dataset with real Wikipedia passages. "
        "Tried Tevatron/wikipedia-nq, Tevatron/wikipedia-trivia, "
        "Tevatron/wikipedia-squad, rajpurkar/squad. Never falling back to "
        "answer-anchor leakage. Errors:\n- " + "\n- ".join(errors)
    )
