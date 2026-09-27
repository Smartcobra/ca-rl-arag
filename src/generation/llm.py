"""Answer generation and query rewrite backends."""

from __future__ import annotations

import gc
import re
from typing import Any

from ..utils import tokenize, truncate

_ANSWER_PREFIXES = ("Answer:", "Final answer:", "A:")
_PURE_ABSTAIN = re.compile(r"^ABSTAIN[.:!?]*$", re.IGNORECASE)
_TRAILING_ABSTAIN = re.compile(r"(?:\s+)ABSTAIN[.:!?]*\s*$", re.IGNORECASE)
_LEADING_ABSTAIN = re.compile(r"^ABSTAIN\b", re.IGNORECASE)
_WH_QUESTION = re.compile(r"^(what|which|who|whom|whose|when|where|why|how)\b", re.IGNORECASE)
_YES_NO_AUX = re.compile(
    r"^(are|is|was|were|do|does|did|can|could|has|have|had)\b",
    re.IGNORECASE,
)
_YES_WORD = re.compile(r"^yes\b", re.IGNORECASE)
_NO_WORD = re.compile(r"^no\b", re.IGNORECASE)
# "is there a name for X" asks for a noun, not yes/no.
_NAME_SEEKING = re.compile(
    r"^(is|are|was|were)\s+there\s+(a|an|any)\s+(name|word|term|title|called)\b",
    re.IGNORECASE,
)
_EXISTENTIAL_THERE = re.compile(r"^(is|are|was|were)\s+there\b", re.IGNORECASE)
_COMPARISON_CUE = re.compile(r"\b(and|both|same)\b", re.IGNORECASE)


def is_yes_no_question(question: str) -> bool:
    """Polar / Hotpot comparison questions only — not name-seeking existentials.

    ``is there a name for the at symbol`` is a span question. ``Are X and Y
    both plants?`` is yes/no.
    """
    q = (question or "").strip()
    if not q or _WH_QUESTION.match(q):
        return False
    if _NAME_SEEKING.match(q):
        return False
    if _EXISTENTIAL_THERE.match(q) and not _COMPARISON_CUE.search(q):
        return False
    return bool(_YES_NO_AUX.match(q))


def parse_yes_no(text: str, allow_abstain: bool = False) -> tuple[str, str]:
    """Normalize a yes/no decode to ``yes`` or ``no``. Never ``Norway`` → ``no``."""
    raw = _strip_answer_prefixes((text or "").strip())
    first = ""
    for line in raw.splitlines():
        stripped = line.strip().strip("\"'")
        if stripped:
            first = stripped
            break
    if first:
        first, _ = _strip_trailing_abstain(first)
        first = first.rstrip(".:!,")
    if first and _YES_WORD.match(first):
        return "yes", "answer"
    if first and _NO_WORD.match(first):
        return "no", "answer"
    if allow_abstain:
        return "ABSTAIN", "abstain"
    return "", "answer"


def should_allow_abstain(question: str, allow_abstain: bool, force_yes_no: bool) -> bool:
    if force_yes_no and is_yes_no_question(question):
        return False
    return bool(allow_abstain)


def parse_answer_or_abstain(
    text: str,
    allow_abstain: bool = True,
    max_chars: int = 200,
) -> tuple[str, str]:
    """Parse generator output as exclusive XOR: a clean span or ABSTAIN, never both.

    Trailing ``ABSTAIN`` is stripped and the remainder is kept as an answer
    (the model answered, then hedged). A leading ``ABSTAIN`` is a refuse.
    The old ``answer.upper()[:20]`` substring check is intentionally gone:
    it missed trailing leaks and kept mixed strings when evidence existed.
    """
    def _abstain() -> tuple[str, str]:
        return ("ABSTAIN", "abstain") if allow_abstain else ("", "answer")

    raw = (text or "").strip()
    if not raw:
        return _abstain()

    raw, _ = _strip_trailing_abstain(raw)
    answer = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            answer = stripped
            break
    answer = _strip_answer_prefixes(answer)
    if not answer:
        return _abstain()
    if _PURE_ABSTAIN.match(answer):
        return _abstain()

    answer, had_trailing = _strip_trailing_abstain(answer)
    if had_trailing:
        if not answer:
            return _abstain()
        return truncate(answer, max_chars), "answer"
    if _LEADING_ABSTAIN.match(answer):
        return _abstain()
    return truncate(answer, max_chars), "answer"


def _strip_answer_prefixes(answer: str) -> str:
    changed = True
    while changed and answer:
        changed = False
        for prefix in _ANSWER_PREFIXES:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix) :].strip()
                changed = True
                break
    return answer


def _strip_trailing_abstain(text: str) -> tuple[str, bool]:
    match = _TRAILING_ABSTAIN.search(text)
    if not match:
        return text, False
    return text[: match.start()].rstrip(), True


class ExtractiveGenerator:
    """Deterministic extractive QA for stable Milestone-2 baselines.

    Prefers short answer spans that appear in evidence. Swap backend later
    without changing the agent API.
    """

    def __init__(
        self,
        max_answer_tokens: int = 64,
        allow_abstain: bool = True,
        force_yes_no: bool = True,
    ):
        self.max_answer_tokens = max_answer_tokens
        self.allow_abstain = bool(allow_abstain)
        self.force_yes_no = bool(force_yes_no)

    def rewrite(self, question: str, evidence: list[dict[str, Any]] | None = None) -> tuple[str, dict[str, int]]:
        extra = []
        if evidence:
            for e in evidence[:3]:
                title = e.get("title") or ""
                for tok in tokenize(title):
                    if tok not in tokenize(question) and tok not in extra:
                        extra.append(tok)
        rewritten = question.strip()
        if extra:
            rewritten = f"{question.strip()} (focus: {' '.join(extra[:6])})"
        else:
            rewritten = re.sub(r"\bwhat is\b", "what exactly is", question.strip(), flags=re.I)
            if rewritten == question.strip():
                rewritten = question.strip() + " key facts"
        return rewritten, {
            "prompt_tokens": max(len(tokenize(question)), 1),
            "completion_tokens": max(len(tokenize(rewritten)), 1),
        }

    def generate(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        allow_abstain: bool | None = None,
    ) -> tuple[str, str, dict[str, int]]:
        allow = self.allow_abstain if allow_abstain is None else allow_abstain
        allow = should_allow_abstain(question, allow, self.force_yes_no)
        prompt_tokens = 200 + sum(len(tokenize(e.get("text", ""))) for e in evidence[:5]) // 4
        if not evidence:
            if allow:
                return "ABSTAIN", "abstain", {"prompt_tokens": 50, "completion_tokens": 2}
            return "", "answer", {"prompt_tokens": 50, "completion_tokens": 2}

        q = question.lower()
        corpus = " ".join(f"{e.get('title', '')}. {e.get('text', '')}" for e in evidence)

        # Pattern-driven extraction for common QA templates
        patterned = self._pattern_answer(q, corpus, evidence)
        if patterned:
            return patterned, "answer", {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max(len(tokenize(patterned)), 1),
            }

        # Title match: if a passage title is a likely short answer and appears relevant
        title_ans = self._title_answer(q, evidence)
        if title_ans:
            return title_ans, "answer", {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max(len(tokenize(title_ans)), 1),
            }

        # Generic short span ranking
        span = self._best_short_span(q, evidence)
        if span:
            return span, "answer", {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max(len(tokenize(span)), 1),
            }

        if allow:
            return "ABSTAIN", "abstain", {"prompt_tokens": prompt_tokens, "completion_tokens": 2}
        return "", "answer", {"prompt_tokens": prompt_tokens, "completion_tokens": 2}

    def _pattern_answer(self, q: str, corpus: str, evidence: list[dict[str, Any]]) -> str | None:
        # "what country has X as its capital" / "capital of X"
        m = re.search(r"country has ([a-z .]+?) as its capital", q)
        if m:
            city = m.group(1).strip()
            m2 = re.search(rf"{re.escape(city)} is the capital of ([A-Z][a-zA-Z]+)", corpus, flags=re.I)
            if m2:
                return m2.group(1)

        m = re.search(r"capital of ([a-z .]+)\??", q)
        if m:
            country = m.group(1).strip().rstrip("?")
            # ignore trailing noise like "today"
            country = re.sub(r"\btoday\b", "", country).strip()
            m2 = re.search(rf"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?) is the capital of {re.escape(country)}", corpus, flags=re.I)
            if m2:
                return m2.group(1)
            m3 = re.search(rf"capital of {re.escape(country)}[^.]*?\b([A-Z][a-zA-Z]+)\b", corpus, flags=re.I)
            if m3:
                return m3.group(1)

        # "is A or B the capital of C" → find which one is
        m = re.search(r"is ([a-z .]+?) or ([a-z .]+?) the capital of ([a-z .]+)", q)
        if m:
            a, b, country = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            for cand in (a, b):
                if re.search(rf"{re.escape(cand)} is the capital of {re.escape(country)}", corpus, flags=re.I):
                    return cand.title() if cand.islower() else cand

        # who / which scientist
        who_patterns = [
            (r"theory of relativity", r"(Albert Einstein)"),
            (r"radioactivity", r"(Marie Curie)"),
            (r"laws of motion|universal gravitation|gravity", r"(Isaac Newton)"),
            (r"analytical engine", r"(Ada Lovelace)"),
            (r"turing", r"(Alan Turing)"),
            (r"relativity", r"(Albert Einstein)"),
        ]
        for qpat, apat in who_patterns:
            if re.search(qpat, q, flags=re.I):
                m2 = re.search(apat, corpus)
                if m2:
                    return m2.group(1)

        # "where is the Eiffel Tower"
        m = re.search(r"where is (?:the )?([a-z0-9 .]+)", q)
        if m:
            thing = m.group(1).strip()
            m2 = re.search(rf"([A-Z][a-zA-Z]+)[^.]*{re.escape(thing)}|{re.escape(thing)}[^.]*([A-Z][a-zA-Z]+)", corpus, flags=re.I)
            if m2:
                return next(g for g in m2.groups() if g)

        # "which capital ..." soft: look for "CITY is the capital"
        if "capital" in q:
            cities = re.findall(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?) is the capital of", corpus)
            if len(set(cities)) == 1:
                return cities[0]
            # pick city whose title/doc overlaps question tokens most
            best, best_s = None, -1
            q_toks = set(tokenize(q))
            for e in evidence:
                title = e.get("title") or ""
                if "is the capital of" in e.get("text", ""):
                    s = len(set(tokenize(title)) & q_toks) + len(set(tokenize(e.get("text", ""))) & q_toks)
                    # prefer titles that are city names appearing as subject
                    if re.search(rf"{re.escape(title)} is the capital of", e.get("text", ""), flags=re.I):
                        s += 5
                    if s > best_s:
                        best_s = s
                        best = title
            if best:
                return best

        return None

    def _title_answer(self, q: str, evidence: list[dict[str, Any]]) -> str | None:
        q_toks = set(tokenize(q))
        best, best_s = None, 0.0
        for e in evidence:
            title = (e.get("title") or "").strip()
            if not title or len(tokenize(title)) > 4:
                continue
            text = e.get("text", "")
            # title should appear as an answer-like entity in text
            if title.lower() not in text.lower() and not text.lower().startswith(title.lower()):
                # still allow if "Title is ..."
                if not re.search(rf"^{re.escape(title)}\b", text):
                    continue
            overlap = len(set(tokenize(text)) & q_toks)
            novelty = 1.0 if title.lower() not in q else 0.0
            score = overlap + 2.0 * novelty + float(e.get("score", 0.0)) * 0.01
            if score > best_s:
                best_s = score
                best = title
        return best

    def _best_short_span(self, q: str, evidence: list[dict[str, Any]]) -> str | None:
        q_toks = set(tokenize(q))
        candidates: list[tuple[float, str]] = []
        for e in evidence:
            text = e.get("text", "")
            # Prefer "X is the capital of Y" objects/subjects
            for m in re.finditer(
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+is the capital of\s+([A-Z][a-z]+)\b",
                text,
            ):
                for span in (m.group(1), m.group(2)):
                    if span.lower() not in q:
                        candidates.append((3.0 + float(e.get("score", 0)) * 0.01, span))
            for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text):
                span = m.group(1)
                if span.lower() in q_toks or span.lower() in q:
                    continue
                if len(span) < 3:
                    continue
                score = 1.0 + 0.2 * len(set(tokenize(text)) & q_toks) + float(e.get("score", 0)) * 0.01
                candidates.append((score, span))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return truncate(candidates[0][1], 80)

    def close(self) -> None:
        """No GPU resources; present so callers can close any generator uniformly."""
        return


class HuggingFaceGenerator:
    """Local instruct-model generator (downloads weights to HF cache, runs on device).

    Intended default: Qwen/Qwen2.5-3B-Instruct. Same generate/rewrite API as extractive.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_answer_tokens: int = 64,
        temperature: float = 0.0,
        device: str = "auto",
        max_input_tokens: int = 2048,
        torch_dtype: str = "auto",
        allow_abstain: bool = True,
        force_yes_no: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "HuggingFace generator requires torch and transformers. "
                "Install with: pip install torch transformers accelerate"
            ) from e

        self.model_name = model_name
        self.max_answer_tokens = max_answer_tokens
        self.temperature = float(temperature)
        self.max_input_tokens = int(max_input_tokens)
        self.allow_abstain = bool(allow_abstain)
        self.force_yes_no = bool(force_yes_no)
        self._torch = torch

        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype(torch_dtype)
        self._closed = False
        self._uses_device_map = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if dtype is not None:
            # Transformers >=4.56 / 5.x: `dtype`. Older releases: `torch_dtype`.
            load_kwargs["dtype"] = dtype
        # device_map="auto" needs accelerate; never combine it with model.to(...).
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            load_kwargs["device_map"] = "auto"
            self._uses_device_map = True
            self.model = self._from_pretrained(AutoModelForCausalLM, model_name, load_kwargs)
        else:
            self.model = self._from_pretrained(AutoModelForCausalLM, model_name, load_kwargs)
            self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def _resolve_device(self, device: str) -> str:
        torch = self._torch
        if device and device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, torch_dtype: str):
        torch = self._torch
        if torch_dtype and torch_dtype != "auto":
            return getattr(torch, torch_dtype)
        if self.device == "cuda":
            return torch.float16
        if self.device == "mps":
            return torch.float16
        return torch.float32

    @staticmethod
    def _from_pretrained(model_cls, model_name: str, load_kwargs: dict[str, Any]):
        try:
            return model_cls.from_pretrained(model_name, **load_kwargs)
        except TypeError as e:
            msg = str(e)
            if "dtype" in load_kwargs and ("dtype" in msg or "unexpected keyword" in msg.lower()):
                fallback = dict(load_kwargs)
                fallback["torch_dtype"] = fallback.pop("dtype")
                return model_cls.from_pretrained(model_name, **fallback)
            raise

    def _input_device(self):
        if getattr(self, "model", None) is None:
            return self.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self.device

    def close(self) -> None:
        """Drop model/tokenizer refs so CUDA memory can be returned to the driver."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        if getattr(self, "tokenizer", None) is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        torch = getattr(self, "_torch", None)
        if torch is None:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def _format_evidence(self, evidence: list[dict[str, Any]], max_passages: int = 5) -> str:
        blocks = []
        for i, e in enumerate(evidence[:max_passages], start=1):
            title = (e.get("title") or "").strip()
            text = truncate(e.get("text") or "", 600)
            head = f"[{i}] {title}".strip() if title else f"[{i}]"
            blocks.append(f"{head}\n{text}")
        return "\n\n".join(blocks) if blocks else "(no evidence)"

    def _chat(self, messages: list[dict[str, str]], max_new_tokens: int) -> tuple[str, dict[str, int]]:
        torch = self._torch
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = self._input_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, prompt_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = int(new_tokens.shape[-1])
        # Drop generate() tensors immediately; results are Python strings/ints only.
        del out, new_tokens, inputs
        return text, {"prompt_tokens": prompt_tokens, "completion_tokens": max(completion_tokens, 1)}

    def rewrite(self, question: str, evidence: list[dict[str, Any]] | None = None) -> tuple[str, dict[str, int]]:
        evidence = evidence or []
        ev = self._format_evidence(evidence, max_passages=3)
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite search queries for retrieval. "
                    "Return only the rewritten query, no quotes or explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question:\n{question}\n\n"
                    f"Current evidence (may be incomplete):\n{ev}\n\n"
                    "Rewritten query:"
                ),
            },
        ]
        text, usage = self._chat(messages, max_new_tokens=min(48, self.max_answer_tokens))
        rewritten = text.splitlines()[0].strip().strip('"').strip("'") if text else question
        if not rewritten:
            rewritten = question
        return rewritten, usage

    def generate(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        allow_abstain: bool | None = None,
    ) -> tuple[str, str, dict[str, int]]:
        allow = self.allow_abstain if allow_abstain is None else allow_abstain
        allow = should_allow_abstain(question, allow, self.force_yes_no)
        ev = self._format_evidence(evidence)
        yes_no = self.force_yes_no and is_yes_no_question(question)
        if yes_no:
            system = (
                "You are a concise open-domain QA assistant. "
                "This is a yes/no question. Reply with exactly yes or no. "
                "Do not reply ABSTAIN. Do not explain or quote. "
                "Use only the evidence. If the passages name the entities in the question, compare them and answer."
            )
            user = f"Evidence:\n{ev}\n\nQuestion: {question}\n\nAnswer (yes or no):"
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            text, usage = self._chat(messages, max_new_tokens=min(8, self.max_answer_tokens))
            answer, mode = parse_yes_no(text, allow_abstain=False)
            return answer, mode, usage
        if allow:
            abstain_rule = (
                "Reply with a short answer span. Never append ABSTAIN after an answer. "
                "Use the single token ABSTAIN only if none of the passages could support any answer "
                "(empty evidence, or passages about clearly unrelated entities). "
                "If a passage title or span could be the answer, you must answer it. "
                "Do not use ABSTAIN as a hedge when you are unsure. "
                "No explanation and no quotes."
            )
        else:
            abstain_rule = (
                "Always give your best short answer from the evidence. "
                "Do not reply with ABSTAIN."
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise open-domain QA assistant. "
                    "Answer using only the provided evidence. "
                    "Reply with a short answer span (a few words when possible). "
                    f"{abstain_rule}"
                ),
            },
            {
                "role": "user",
                "content": f"Evidence:\n{ev}\n\nQuestion: {question}\n\nAnswer:",
            },
        ]
        text, usage = self._chat(messages, max_new_tokens=self.max_answer_tokens)
        answer, mode = parse_answer_or_abstain(text, allow_abstain=allow)
        return answer, mode, usage


def build_generator(cfg: dict[str, Any]):
    gcfg = cfg.get("generation", {})
    backend = gcfg.get("backend", "extractive")
    max_tok = int(gcfg.get("max_answer_tokens", 64))
    allow_abstain = bool(gcfg.get("allow_abstain", True))
    force_yes_no = bool(gcfg.get("force_yes_no", True))
    if backend == "extractive":
        return ExtractiveGenerator(
            max_answer_tokens=max_tok,
            allow_abstain=allow_abstain,
            force_yes_no=force_yes_no,
        )
    if backend in {"huggingface", "hf", "qwen"}:
        model_name = gcfg.get("model_name") or "Qwen/Qwen2.5-3B-Instruct"
        return HuggingFaceGenerator(
            model_name=model_name,
            max_answer_tokens=max_tok,
            temperature=float(gcfg.get("temperature", 0.0)),
            device=str(gcfg.get("device", "auto")),
            max_input_tokens=int(gcfg.get("max_input_tokens", 2048)),
            torch_dtype=str(gcfg.get("torch_dtype", "auto")),
            allow_abstain=allow_abstain,
            force_yes_no=force_yes_no,
        )
    if backend == "openai":
        raise NotImplementedError(
            "OpenAI generation backend is not wired yet. "
            "Use generation.backend: huggingface with Qwen/Qwen2.5-3B-Instruct, or extractive."
        )
    raise ValueError(f"Unknown generation backend: {backend}")
