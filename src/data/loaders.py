"""Dataset loading and slice preparation for NQ + HotpotQA."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from ..utils import write_jsonl

HOTPOT_SLICE_SOURCE = "hotpot_qa"
HOTPOT_DISTRACTOR_SOURCE = "hotpot_distractor"
NQ_ANCHOR_SOURCE = "nq_anchor"  # legacy leak; HF prepare must never write this
NQ_WIKI_SOURCE = "nq_kilt_dpr"
NQ_WIKI_NEG_SOURCE = "nq_wiki_distractor"


def _stable_id(*parts: str) -> str:
    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return h


def _hotpot_gold_titles(row: dict[str, Any]) -> set[str]:
    titles = row.get("supporting_facts", {}).get("title", []) or []
    return {t for t in titles if t}


def passages_from_hotpot_row(
    row: dict[str, Any],
    *,
    as_distractor: bool = False,
) -> list[dict[str, Any]]:
    """Turn one Hotpot context into passages.

    Slice rows keep supporting-fact gold flags. Unused pool rows are always
    distractors: their supporting facts belong to questions we did not keep.
    """
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    gold_titles = set() if as_distractor else _hotpot_gold_titles(row)
    source = HOTPOT_DISTRACTOR_SOURCE if as_distractor else HOTPOT_SLICE_SOURCE
    passages: list[dict[str, Any]] = []
    for title, sents in zip(titles, sentences):
        text = " ".join(sents).strip()
        if not text:
            continue
        passages.append(
            {
                "passage_id": _stable_id("hotpot", title, text[:80]),
                "title": title,
                "text": text,
                "source": source,
                "is_gold_support": (not as_distractor) and (title in gold_titles),
            }
        )
    return passages


def add_hotpot_distractor_pool(
    existing: list[dict[str, Any]],
    unused_rows: Iterable[dict[str, Any]],
    target_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Grow a corpus toward ``target_size`` using unused Hotpot contexts.

    Existing slice / NQ Wikipedia passages are never rewritten. Dedup keeps the
    first copy (the gold/slice row). ``target_size <= 0`` or already met is a
    no-op. Stops as soon as ``len(corpus) >= target_size``.
    """
    by_id = {p["passage_id"]: p for p in existing}
    n_before = len(by_id)
    n_added = 0
    n_dup = 0
    n_rows = 0
    target = int(target_size)
    if target <= n_before:
        return list(by_id.values()), {
            "n_passages": n_before,
            "n_slice_passages": n_before,
            "n_distractors_added": 0,
            "n_distractor_rows_scanned": 0,
            "n_distractor_dups_skipped": 0,
            "target": target,
            "hit_target": target > 0 and n_before >= target,
        }

    for row in unused_rows:
        if len(by_id) >= target:
            break
        n_rows += 1
        for passage in passages_from_hotpot_row(row, as_distractor=True):
            pid = passage["passage_id"]
            if pid in by_id:
                n_dup += 1
                continue
            by_id[pid] = passage
            n_added += 1
            if len(by_id) >= target:
                break

    passages = list(by_id.values())
    return passages, {
        "n_passages": len(passages),
        "n_slice_passages": n_before,
        "n_distractors_added": n_added,
        "n_distractor_rows_scanned": n_rows,
        "n_distractor_dups_skipped": n_dup,
        "target": target,
        "hit_target": len(passages) >= target,
    }


def corpus_source_counts(passages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {
        "n_passages": len(passages),
        "n_gold_support": 0,
        "n_hotpot_slice": 0,
        "n_hotpot_distractor": 0,
        "n_nq_wiki": 0,
        "n_nq_wiki_neg": 0,
        "n_nq_anchor": 0,
    }
    for p in passages:
        if p.get("is_gold_support"):
            counts["n_gold_support"] += 1
        src = p.get("source", "")
        if src == HOTPOT_SLICE_SOURCE:
            counts["n_hotpot_slice"] += 1
        elif src == HOTPOT_DISTRACTOR_SOURCE:
            counts["n_hotpot_distractor"] += 1
        elif src == NQ_WIKI_SOURCE:
            counts["n_nq_wiki"] += 1
        elif src == NQ_WIKI_NEG_SOURCE:
            counts["n_nq_wiki_neg"] += 1
        elif src == NQ_ANCHOR_SOURCE:
            counts["n_nq_anchor"] += 1
    return counts


def hotpot_to_examples(raw_rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    seen_passages: set[str] = set()

    for i, row in enumerate(raw_rows):
        gold_titles = _hotpot_gold_titles(row)
        row_passages = passages_from_hotpot_row(row, as_distractor=False)
        local_passage_ids = [p["passage_id"] for p in row_passages]
        for passage in row_passages:
            if passage["passage_id"] not in seen_passages:
                seen_passages.add(passage["passage_id"])
                passages.append(passage)

        examples.append(
            {
                "id": f"hotpot_{split}_{row.get('id', i)}",
                "dataset": "hotpot_qa",
                "split": split,
                "question": row["question"],
                "answers": [row["answer"]],
                "answer": row["answer"],
                "type": row.get("type"),
                "level": row.get("level"),
                "supporting_titles": sorted(gold_titles),
                "local_passage_ids": local_passage_ids,
            }
        )
    return examples, passages


def nq_to_examples(raw_rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Questions-only converter (no evidence). HF ranking data must use wiki_passages."""
    examples: list[dict[str, Any]] = []
    for i, row in enumerate(raw_rows):
        answers = row.get("answer") or row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        answers = [a for a in answers if a]
        if not answers:
            continue
        examples.append(
            {
                "id": f"nq_{split}_{i}",
                "dataset": "natural_questions",
                "split": split,
                "question": row["question"],
                "answers": answers,
                "answer": answers[0],
                "type": "single_hop",
                "level": None,
                "supporting_titles": [],
                "local_passage_ids": [],
            }
        )
    return examples


def build_synthetic_pilot(
    n_train_hotpot: int = 20,
    n_train_nq: int = 15,
    n_eval_hotpot: int = 10,
    n_eval_nq: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Offline-friendly slice with known answers in-corpus (for smoke tests / no-network)."""
    facts = [
        ("Paris", "France", "Paris is the capital of France and home to the Eiffel Tower."),
        ("Berlin", "Germany", "Berlin is the capital of Germany and was divided during the Cold War."),
        ("Rome", "Italy", "Rome is the capital of Italy and was the center of the Roman Empire."),
        ("Madrid", "Spain", "Madrid is the capital of Spain and hosts the Prado Museum."),
        ("Lisbon", "Portugal", "Lisbon is the capital of Portugal on the Atlantic coast."),
        ("Ottawa", "Canada", "Ottawa is the capital of Canada, not Toronto."),
        ("Canberra", "Australia", "Canberra is the capital of Australia, not Sydney."),
        ("Brasilia", "Brazil", "Brasilia is the capital of Brazil, planned in the 1950s."),
        ("Cairo", "Egypt", "Cairo is the capital of Egypt near the Nile River."),
        ("Tokyo", "Japan", "Tokyo is the capital of Japan and one of the largest metro areas."),
        ("New Delhi", "India", "New Delhi is the capital of India."),
        ("Beijing", "China", "Beijing is the capital of China."),
        ("Nairobi", "Kenya", "Nairobi is the capital of Kenya in East Africa."),
        ("Ankara", "Turkey", "Ankara is the capital of Turkey, while Istanbul is the largest city."),
        ("Vienna", "Austria", "Vienna is the capital of Austria and birthplace of many composers."),
    ]
    scientists = [
        ("Albert Einstein", "relativity", "Albert Einstein developed the theory of relativity."),
        ("Marie Curie", "radioactivity", "Marie Curie pioneered research on radioactivity and won two Nobel Prizes."),
        ("Isaac Newton", "gravity", "Isaac Newton formulated the laws of motion and universal gravitation."),
        ("Ada Lovelace", "computing", "Ada Lovelace wrote early notes on Charles Babbage's Analytical Engine."),
        ("Alan Turing", "computation", "Alan Turing founded concepts of modern computation and Turing machines."),
    ]

    passages: list[dict[str, Any]] = []
    for city, country, text in facts:
        passages.append(
            {
                "passage_id": _stable_id("syn", city, country),
                "title": city,
                "text": text,
                "source": "synthetic",
                "is_gold_support": True,
            }
        )
    for person, topic, text in scientists:
        passages.append(
            {
                "passage_id": _stable_id("syn", person, topic),
                "title": person,
                "text": text,
                "source": "synthetic",
                "is_gold_support": True,
            }
        )
    passages.append(
        {
            "passage_id": _stable_id("syn", "distractor", "1"),
            "title": "Tourism notes",
            "text": "Many capital cities attract tourists with museums, parks, and historic districts.",
            "source": "synthetic",
            "is_gold_support": False,
        }
    )

    multi = [
        ("What country has Paris as its capital?", "France", ["Paris"]),
        ("What country has Berlin as its capital?", "Germany", ["Berlin"]),
        ("What country has Rome as its capital?", "Italy", ["Rome"]),
        ("What country has Madrid as its capital?", "Spain", ["Madrid"]),
        ("What country has Lisbon as its capital?", "Portugal", ["Lisbon"]),
        ("What country has Ottawa as its capital?", "Canada", ["Ottawa"]),
        ("What country has Canberra as its capital?", "Australia", ["Canberra"]),
        ("What country has Brasilia as its capital?", "Brazil", ["Brasilia"]),
        ("What country has Cairo as its capital?", "Egypt", ["Cairo"]),
        ("What country has Tokyo as its capital?", "Japan", ["Tokyo"]),
        ("What country has New Delhi as its capital?", "India", ["New Delhi"]),
        ("What country has Beijing as its capital?", "China", ["Beijing"]),
        ("Who developed the theory of relativity?", "Albert Einstein", ["Albert Einstein"]),
        ("Who pioneered research on radioactivity?", "Marie Curie", ["Marie Curie"]),
        ("Who formulated the laws of motion and universal gravitation?", "Isaac Newton", ["Isaac Newton"]),
        ("Who wrote early notes on the Analytical Engine?", "Ada Lovelace", ["Ada Lovelace"]),
        ("Who founded concepts of modern computation and Turing machines?", "Alan Turing", ["Alan Turing"]),
        ("Is Ottawa or Toronto the capital of Canada?", "Ottawa", ["Ottawa"]),
        ("Is Canberra or Sydney the capital of Australia?", "Canberra", ["Canberra"]),
        ("Is Ankara or Istanbul the capital of Turkey?", "Ankara", ["Ankara"]),
        ("Where is the Eiffel Tower located?", "Paris", ["Paris"]),
        ("Which capital was planned in the 1950s in Brazil?", "Brasilia", ["Brasilia"]),
        ("Which capital is near the Nile River?", "Cairo", ["Cairo"]),
        ("Which scientist won two Nobel Prizes for radioactivity research?", "Marie Curie", ["Marie Curie"]),
        ("Which city is the capital of Austria?", "Vienna", ["Vienna"]),
        ("Which city is the capital of Kenya?", "Nairobi", ["Nairobi"]),
        ("What is the capital of France?", "Paris", ["Paris"]),
        ("What is the capital of Germany?", "Berlin", ["Berlin"]),
        ("What is the capital of Italy?", "Rome", ["Rome"]),
        ("What is the capital of Spain?", "Madrid", ["Madrid"]),
    ]
    single = [
        ("What is the capital of Portugal?", "Lisbon"),
        ("What is the capital of Canada?", "Ottawa"),
        ("What is the capital of Australia?", "Canberra"),
        ("What is the capital of Brazil?", "Brasilia"),
        ("What is the capital of Egypt?", "Cairo"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of India?", "New Delhi"),
        ("What is the capital of China?", "Beijing"),
        ("What is the capital of Kenya?", "Nairobi"),
        ("What is the capital of Turkey?", "Ankara"),
        ("What is the capital of Austria?", "Vienna"),
        ("Who developed relativity?", "Albert Einstein"),
        ("Who researched radioactivity?", "Marie Curie"),
        ("Who formulated gravity laws?", "Isaac Newton"),
        ("Who is associated with the Analytical Engine notes?", "Ada Lovelace"),
        ("Who introduced Turing machines?", "Alan Turing"),
        ("Where is the Prado Museum?", "Madrid"),
        ("Which capital city is on the Atlantic coast of Portugal?", "Lisbon"),
        ("Which capital was divided during the Cold War?", "Berlin"),
        ("Which capital was the center of the Roman Empire?", "Rome"),
    ]

    # Hold out last N for eval so train/eval are disjoint; cap so train is non-empty.
    n_eval_hotpot = min(n_eval_hotpot, max(0, len(multi) // 3))
    n_eval_nq = min(n_eval_nq, max(0, len(single) // 3))
    hotpot_eval_src = multi[-n_eval_hotpot:] if n_eval_hotpot else []
    hotpot_train_pool = multi[: len(multi) - n_eval_hotpot] if n_eval_hotpot else multi
    hotpot_train_src = hotpot_train_pool[:n_train_hotpot]
    nq_eval_src = single[-n_eval_nq:] if n_eval_nq else []
    nq_train_pool = single[: len(single) - n_eval_nq] if n_eval_nq else single
    nq_train_src = nq_train_pool[:n_train_nq]

    train: list[dict[str, Any]] = []
    eval_set: list[dict[str, Any]] = []

    for i, (q, ans, titles) in enumerate(hotpot_train_src):
        train.append(
            {
                "id": f"syn_hotpot_train_{i}",
                "dataset": "hotpot_qa",
                "split": "train",
                "question": q,
                "answers": [ans],
                "answer": ans,
                "type": "bridge",
                "level": "easy",
                "supporting_titles": titles,
                "local_passage_ids": [],
            }
        )
    for i, (q, ans, titles) in enumerate(hotpot_eval_src):
        eval_set.append(
            {
                "id": f"syn_hotpot_eval_{i}",
                "dataset": "hotpot_qa",
                "split": "eval",
                "question": q,
                "answers": [ans],
                "answer": ans,
                "type": "bridge",
                "level": "easy",
                "supporting_titles": titles,
                "local_passage_ids": [],
            }
        )
    for i, (q, ans) in enumerate(nq_train_src):
        train.append(
            {
                "id": f"syn_nq_train_{i}",
                "dataset": "natural_questions",
                "split": "train",
                "question": q,
                "answers": [ans],
                "answer": ans,
                "type": "single_hop",
                "level": None,
                "supporting_titles": [],
                "local_passage_ids": [],
            }
        )
    for i, (q, ans) in enumerate(nq_eval_src):
        eval_set.append(
            {
                "id": f"syn_nq_eval_{i}",
                "dataset": "natural_questions",
                "split": "eval",
                "question": q,
                "answers": [ans],
                "answer": ans,
                "type": "single_hop",
                "level": None,
                "supporting_titles": [],
                "local_passage_ids": [],
            }
        )

    return train, eval_set, passages


def save_processed(
    train: list[dict[str, Any]],
    eval_set: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    out_dir: str,
) -> dict[str, str]:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": str(out / "train_slice.jsonl"),
        "eval": str(out / "eval_slice.jsonl"),
        "corpus": str(out / "corpus.jsonl"),
    }
    write_jsonl(paths["train"], train)
    write_jsonl(paths["eval"], eval_set)
    write_jsonl(paths["corpus"], passages)
    return paths


def stratified_limit(
    examples: list[dict[str, Any]],
    limit: int | None,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Take ``limit`` examples while preserving the dataset mix.

    Prefix ``examples[:limit]`` on a hotpot-first JSONL rewrites the mix
    (e.g. 150+150 with ``--limit 40`` becomes 40 Hotpot + 0 NQ). This keeps
    approximately ``round(limit * n_ds / n_total)`` from each ``dataset``,
    then adjusts leftover rounding so ``len(out) == limit`` (or all rows if
    fewer exist). Within each dataset, original JSONL order is kept (HF
    validation prefix). ``seed`` is reserved for leftover-tie stability and
    is currently unused because ties break by dataset name order.
    """
    del seed  # documented; leftover ties use dataset display order
    if not examples or limit is None or limit <= 0 or limit >= len(examples):
        return list(examples)

    from ..metrics import ordered_dataset_names

    groups: dict[str, list[dict[str, Any]]] = {}
    for ex in examples:
        ds = str(ex.get("dataset") or "unknown")
        groups.setdefault(ds, []).append(ex)

    n_total = len(examples)
    names = ordered_dataset_names(groups)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for ds in names:
        exact = limit * len(groups[ds]) / n_total
        floor = min(int(exact), len(groups[ds]))
        quotas[ds] = floor
        remainders.append((exact - int(exact), ds))

    leftover = min(limit, n_total) - sum(quotas.values())
    remainders.sort(key=lambda item: (-item[0], names.index(item[1])))
    for _, ds in remainders:
        if leftover <= 0:
            break
        if quotas[ds] < len(groups[ds]):
            quotas[ds] += 1
            leftover -= 1
    if leftover > 0:
        for ds in names:
            if leftover <= 0:
                break
            spare = len(groups[ds]) - quotas[ds]
            take = min(spare, leftover)
            quotas[ds] += take
            leftover -= take

    keep = {id(ex) for ds in names for ex in groups[ds][: quotas[ds]]}
    return [ex for ex in examples if id(ex) in keep]
