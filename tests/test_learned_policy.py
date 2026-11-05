#!/usr/bin/env python3
"""Unit tests for the tiny REINFORCE policy. No GPU, no eval slice."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agentic_rag import ACTIONS
from src.policies import get_policy
from src.policies.learned import (
    MAX_HIDDEN,
    MAX_PARAM_COUNT,
    OBS_DIM,
    LearnedPolicy,
    PolicyMLP,
    assert_train_only_path,
    legal_mask,
    learned_checkpoint_path,
    parameter_count,
)
from src.rag_env import ACTION_TO_IDX, standardize_score, vectorize_structured_obs

_AGENT = {
    "max_steps": 8,
    "max_retrieve": 3,
    "max_rewrite": 2,
    "max_rerank": 2,
    "max_verify": 2,
}
_CFG = {"agent": _AGENT}


def _empty_evidence_obs(*, remaining_frac: float = 1.0) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[2] = remaining_frac
    return obs


def test_mlp_shape_and_size() -> None:
    mlp = PolicyMLP(16)
    n = parameter_count(mlp)
    assert OBS_DIM == 25
    assert n == 501
    assert n <= MAX_PARAM_COUNT
    try:
        PolicyMLP(32)
    except ValueError as exc:
        assert "720" in str(exc)
    else:
        raise AssertionError("width 32 is 997 parameters and must exceed the 720 cap")
    out = mlp(torch.randn(4, OBS_DIM))
    assert tuple(out.shape) == (4, len(ACTIONS))


def test_hidden_cap() -> None:
    try:
        PolicyMLP(MAX_HIDDEN + 1)
    except ValueError as exc:
        assert str(MAX_HIDDEN) in str(exc)
        return
    raise AssertionError("expected ValueError for oversized hidden")


def test_legal_mask_empty_evidence() -> None:
    mask = legal_mask(_empty_evidence_obs(), _CFG)
    assert mask[ACTION_TO_IDX["retrieve"]]
    assert mask[ACTION_TO_IDX["rewrite"]]
    assert not mask[ACTION_TO_IDX["rerank"]]
    assert not mask[ACTION_TO_IDX["verify"]]
    assert not mask[ACTION_TO_IDX["stop"]]


def test_legal_mask_last_step() -> None:
    mask = legal_mask(_empty_evidence_obs(remaining_frac=1.0 / 8.0), _CFG)
    assert mask[ACTION_TO_IDX["stop"]]
    assert mask.sum() == 1


def _evidence_obs(*, remaining_frac: float = 1.0, n_verify: float = 0.0) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[1] = 0.5
    obs[2] = remaining_frac
    obs[6] = 1.0 / 3.0
    obs[9] = n_verify / 2.0
    return obs


def test_force_first_retrieve_mask() -> None:
    forced = {"agent": {**_AGENT, "force_first_retrieve": True}}
    mask = legal_mask(_empty_evidence_obs(), forced)
    assert mask[ACTION_TO_IDX["retrieve"]]
    assert int(mask.sum()) == 1
    open_mask = legal_mask(_empty_evidence_obs(), _CFG)
    assert open_mask[ACTION_TO_IDX["retrieve"]]
    assert open_mask[ACTION_TO_IDX["rewrite"]]
    last = legal_mask(_empty_evidence_obs(remaining_frac=1.0 / 8.0), forced)
    assert last[ACTION_TO_IDX["stop"]]
    assert int(last.sum()) == 1
    seen = _empty_evidence_obs()
    seen[1] = 0.5
    seen[6] = 1.0 / 3.0
    after = legal_mask(seen, forced)
    assert after[ACTION_TO_IDX["retrieve"]]
    assert after[ACTION_TO_IDX["rewrite"]]
    assert after[ACTION_TO_IDX["stop"]]


def test_inert_mask_blocks_repeated_retrieve_second_verify_and_last_rewrite() -> None:
    blocked = {"agent": {**_AGENT, "force_first_retrieve": True, "block_inert_actions": True}}
    seen = _evidence_obs()
    # The flag is off: a query already searched is still a legal retrieve.
    open_mask = legal_mask(seen, {"agent": {**_AGENT, "block_inert_actions": False}}, query_already_searched=True)
    assert open_mask[ACTION_TO_IDX["retrieve"]]
    same_query = legal_mask(seen, blocked, query_already_searched=True)
    assert not same_query[ACTION_TO_IDX["retrieve"]]
    assert same_query[ACTION_TO_IDX["rewrite"]]
    assert same_query[ACTION_TO_IDX["verify"]]
    fresh_query = legal_mask(seen, blocked, query_already_searched=False)
    assert fresh_query[ACTION_TO_IDX["retrieve"]]
    second_verify = legal_mask(_evidence_obs(n_verify=1.0), blocked, query_already_searched=False)
    assert not second_verify[ACTION_TO_IDX["verify"]]
    first_verify = legal_mask(_evidence_obs(n_verify=0.0), blocked, query_already_searched=False)
    assert first_verify[ACTION_TO_IDX["verify"]]
    # Two steps left: this action, then the forced stop. A rewrite cannot be retrieved.
    last_slot = legal_mask(_evidence_obs(remaining_frac=2.0 / 8.0), blocked, query_already_searched=False)
    assert not last_slot[ACTION_TO_IDX["rewrite"]]
    assert last_slot[ACTION_TO_IDX["retrieve"]]
    assert last_slot[ACTION_TO_IDX["stop"]]
    earlier = legal_mask(_evidence_obs(remaining_frac=3.0 / 8.0), blocked, query_already_searched=False)
    assert earlier[ACTION_TO_IDX["rewrite"]]
    # Same query, last slot, verify already used: stop is the fallback.
    stuck = legal_mask(
        _evidence_obs(remaining_frac=2.0 / 8.0, n_verify=1.0),
        blocked,
        query_already_searched=True,
    )
    assert stuck[ACTION_TO_IDX["stop"]]
    assert not stuck[ACTION_TO_IDX["retrieve"]]
    assert not stuck[ACTION_TO_IDX["rewrite"]]
    assert not stuck[ACTION_TO_IDX["verify"]]
    empty = _empty_evidence_obs(remaining_frac=2.0 / 8.0)
    empty[6] = 1.0 / 3.0
    empty_last = legal_mask(empty, blocked, query_already_searched=True)
    assert empty_last[ACTION_TO_IDX["stop"]]
    assert int(empty_last.sum()) == 1
    # Step 0 stays retrieve. The inert rules do not open the first action.
    first = legal_mask(_empty_evidence_obs(), blocked, query_already_searched=False)
    assert first[ACTION_TO_IDX["retrieve"]]
    assert int(first.sum()) == 1


def test_act_respects_inert_mask() -> None:
    cfg = {"agent": {**_AGENT, "block_inert_actions": True}}
    pol = LearnedPolicy(hidden=16, seed=1)
    obs = _evidence_obs(remaining_frac=2.0 / 8.0, n_verify=1.0)
    for _ in range(30):
        action, logp, ent = pol.act(obs, cfg=cfg, query_already_searched=True)
        assert action in {ACTION_TO_IDX["rerank"], ACTION_TO_IDX["stop"]}
        assert torch.isfinite(logp)
        assert torch.isfinite(ent)


def test_retrieved_query_is_marked_searched() -> None:
    from src.agentic_rag import AgentState, AgenticRAG
    from src.cost import CostTracker
    from src.rewards import RewardComputer

    tracker = CostTracker({})
    state = AgentState(question="q", current_query="q")
    cfg = {"agent": {"max_steps": 8}, "budget": {"max_usd": 0.05}}
    assert state.observation(tracker, cfg)["query_already_searched"] is False

    class _Retriever:
        def search(self, query: str, top_k: int = 5):
            assert query == "q"
            return [{"passage_id": "p", "title": "t", "text": "body", "score": 1.0}]

    agent = AgenticRAG(
        {
            "retrieval": {},
            "verification": {"backend": "lexical_nli"},
            "agent": {"max_steps": 8},
            "budget": {},
            "price_card": {"retrieval": {"bm25_per_call": 5e-5}},
        },
        _Retriever(),
        reward_computer=RewardComputer({}),
        generator=object(),
    )
    state, _info = agent.step(state, "retrieve", CostTracker(agent.cfg["price_card"]))
    assert "q" in state.retrieved_queries
    assert state.observation(tracker, cfg)["query_already_searched"] is True
    state.current_query = "rewritten q"
    assert state.observation(tracker, cfg)["query_already_searched"] is False


def test_same_seed_same_weights_despite_global_rng() -> None:
    torch.manual_seed(1)
    first = LearnedPolicy(hidden=16, seed=42)
    torch.manual_seed(99)
    second = LearnedPolicy(hidden=16, seed=42)
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)


def test_sample_uses_private_generator() -> None:
    pol = LearnedPolicy(hidden=16, seed=0)
    obs = _empty_evidence_obs()
    pol.reseed_sampler(123)
    actions_a = [pol.act(obs, cfg=_CFG)[0] for _ in range(8)]
    pol.reseed_sampler(123)
    torch.manual_seed(7)
    expected = torch.random.get_rng_state().clone()
    actions_b = [pol.act(obs, cfg=_CFG)[0] for _ in range(8)]
    assert actions_a == actions_b
    assert torch.equal(torch.random.get_rng_state(), expected)


def test_act_deterministic_is_argmax() -> None:
    pol = LearnedPolicy(hidden=16, seed=0)
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[1] = 0.5
    obs[2] = 1.0
    with torch.no_grad():
        a1, _, _ = pol.act(obs, cfg=_CFG, deterministic=True)
        a2, _, _ = pol.act(obs, cfg=_CFG, deterministic=True)
        expected = int(pol._dist(obs, _CFG).probs.argmax(dim=-1).item())
    assert a1 == a2 == expected


def test_act_respects_mask() -> None:
    pol = LearnedPolicy(hidden=16, seed=0)
    obs = _empty_evidence_obs()
    for _ in range(20):
        action, logp, ent = pol.act(obs, cfg=_CFG)
        assert action in {ACTION_TO_IDX["retrieve"], ACTION_TO_IDX["rewrite"]}
        assert torch.isfinite(logp)
        assert torch.isfinite(ent)


def test_reinforce_step_finite() -> None:
    pol = LearnedPolicy(hidden=16, seed=0)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-2)
    obs = _empty_evidence_obs()
    _action, logp, ent = pol.act(obs, cfg=_CFG)
    loss = -(logp * 1.0) - 0.01 * ent
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item()


def test_verify_features_and_no_dataset_id() -> None:
    cfg = {"agent": {"max_steps": 8}, "budget": {"max_usd": 0.05}}
    base = {
        "mean_score": 10.0,
        "n_evidence": 2,
        "remaining_steps": 6,
        "remaining_usd": 0.04,
        "top_scores": [20.0, 5.0],
        "counts": {"retrieve": 1, "rewrite": 0, "rerank": 0, "verify": 1},
        "verification": {"support": 0.2, "contradiction": 0.1, "label": "support"},
    }
    support = vectorize_structured_obs(base, cfg)
    contra = vectorize_structured_obs(
        {**base, "verification": {"support": 0.2, "contradiction": 0.1, "label": "contradiction"}},
        cfg,
    )
    neutral = vectorize_structured_obs(
        {**base, "verification": {"support": 0.2, "contradiction": 0.1, "label": "neutral"}},
        cfg,
    )
    before = vectorize_structured_obs({**base, "verification": None, "top_scores": []}, cfg)
    assert tuple(support.shape) == (OBS_DIM,)
    assert np.allclose(support[:12], contra[:12])
    assert np.isclose(support[15], contra[15])
    assert list(support[12:15]) == [1.0, 0.0, 0.0]
    assert list(contra[12:15]) == [0.0, 1.0, 0.0]
    assert list(neutral[12:15]) == [0.0, 0.0, 1.0]
    assert list(before[12:15]) == [0.0, 0.0, 0.0]
    assert before[10] == 0.0 and before[11] == 0.0 and before[15] == 0.0
    assert np.isclose(support[0], np.float32(standardize_score(10.0, "mean")))
    assert np.isclose(support[10], np.float32(standardize_score(20.0, "top1")))
    assert np.isclose(support[11], np.float32(standardize_score(15.0, "gap")))
    assert np.isclose(support[15], np.float32(standardize_score(5.0, "top5_min")))
    assert abs(float(support[10]) - float(support[11])) > 0.05
    tagged = dict(base)
    tagged["dataset"] = "hotpot_qa"
    assert np.allclose(support, vectorize_structured_obs(tagged, cfg))
    one = vectorize_structured_obs({**base, "top_scores": [8.0]}, cfg)
    assert one[11] == 0.0
    assert np.isclose(one[10], np.float32(standardize_score(8.0, "top1")))
    assert np.isclose(one[15], np.float32(standardize_score(8.0, "top5_min")))
    assert abs(float(one[10]) - float(support[10])) > 0.05
    assert np.all(support[16:] == 0.0)


def test_question_and_draft_features() -> None:
    cfg = {"agent": {"max_steps": 8}, "budget": {"max_usd": 0.05}}
    who = vectorize_structured_obs({"query": "Who is Barack Obama?"}, cfg)
    assert tuple(who.shape) == (OBS_DIM,)
    assert np.isclose(who[16], np.float32(4.0 / 32.0))
    assert list(who[17:22]) == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert np.isclose(who[22], np.float32(2.0 / 5.0))
    assert who[23] == 0.0 and who[24] == 0.0
    yes_no = vectorize_structured_obs({"query": "Are cats and dogs both animals?"}, cfg)
    assert list(yes_no[17:22]) == [0.0, 0.0, 0.0, 0.0, 1.0]
    other = vectorize_structured_obs({"query": "Where is Paris?"}, cfg)
    assert list(other[17:22]) == [0.0, 0.0, 0.0, 0.0, 0.0]
    tagged = vectorize_structured_obs(
        {"query": "Who is Barack Obama?", "dataset": "hotpot_qa", "current_query": "rewritten query about something else"},
        cfg,
    )
    assert np.allclose(who, tagged)
    passage = "Barack Obama was born in Hawaii."
    abstain = vectorize_structured_obs(
        {"query": "Who is Barack Obama?", "draft_answer_full": "ABSTAIN", "top_passage_text": passage},
        cfg,
    )
    assert abstain[23] == 1.0 and abstain[24] == 0.0
    overlap = vectorize_structured_obs(
        {"query": "Who is Barack Obama?", "draft_answer_full": "born in Hawaii", "top_passage_text": passage},
        cfg,
    )
    assert overlap[23] == 0.0
    assert float(overlap[24]) > 0.0
    empty = vectorize_structured_obs(
        {"query": "Who is Barack Obama?", "draft_answer_full": "", "top_passage_text": passage},
        cfg,
    )
    assert empty[23] == 0.0 and empty[24] == 0.0


def test_assert_train_only_path() -> None:
    p = assert_train_only_path("data/processed/train_slice.jsonl")
    assert p.name == "train_slice.jsonl"
    valid = assert_train_only_path("data/processed/valid_slice.jsonl")
    assert valid.name == "valid_slice.jsonl"
    try:
        assert_train_only_path("data/processed/eval_slice.jsonl")
    except ValueError as exc:
        assert "eval" in str(exc).lower()
        return
    raise AssertionError("expected ValueError for eval_slice.jsonl")


def test_as_callable_empty_evidence() -> None:
    pol = LearnedPolicy(hidden=8, seed=0)
    fn = pol.as_callable(_CFG, deterministic=True)
    structured = {
        "mean_score": 0.0,
        "n_evidence": 0,
        "remaining_steps": 8,
        "remaining_usd": 0.05,
        "verification": None,
        "counts": {"retrieve": 0, "rewrite": 0, "rerank": 0, "verify": 0},
    }
    action = fn(structured, None)
    assert action in {"retrieve", "rewrite"}


def test_get_policy_learned_requires_cfg() -> None:
    try:
        get_policy("learned")
    except ValueError as exc:
        assert "cfg" in str(exc).lower()
        return
    raise AssertionError("expected ValueError when cfg is missing")


def test_learned_checkpoint_path_default() -> None:
    cfg = {"root": str(ROOT), "policy": {"learned": {"checkpoint": "results/checkpoints/learned_policy.pt"}}}
    path = learned_checkpoint_path(cfg)
    assert path.name == "learned_policy.pt"
    assert "eval" not in path.name


def test_save_load_roundtrip(tmp_path=None) -> None:
    if tmp_path is None:
        dest = ROOT / "results" / "checkpoints" / "_test_learned.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cleanup = True
    else:
        dest = Path(tmp_path) / "learned.pt"
        cleanup = False
    pol = LearnedPolicy(hidden=8, seed=7)
    pol.save(dest, extra={"epoch": 1, "mean_reward": 0.1})
    loaded = LearnedPolicy.load(dest)
    assert loaded.hidden == 8
    x = torch.zeros(1, OBS_DIM)
    assert torch.allclose(pol.mlp(x), loaded.mlp(x))
    if cleanup and dest.exists():
        dest.unlink()


def main() -> None:
    test_mlp_shape_and_size()
    test_hidden_cap()
    test_legal_mask_empty_evidence()
    test_legal_mask_last_step()
    test_force_first_retrieve_mask()
    test_inert_mask_blocks_repeated_retrieve_second_verify_and_last_rewrite()
    test_act_respects_inert_mask()
    test_retrieved_query_is_marked_searched()
    test_act_respects_mask()
    test_same_seed_same_weights_despite_global_rng()
    test_sample_uses_private_generator()
    test_act_deterministic_is_argmax()
    test_reinforce_step_finite()
    test_verify_features_and_no_dataset_id()
    test_question_and_draft_features()
    test_assert_train_only_path()
    test_as_callable_empty_evidence()
    test_get_policy_learned_requires_cfg()
    test_learned_checkpoint_path_default()
    test_save_load_roundtrip()
    print("LEARNED POLICY OK")


if __name__ == "__main__":
    main()
