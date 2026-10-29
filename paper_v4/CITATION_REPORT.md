# Citation report

Each line is a key in `paper_v4/references.bib`, the link that was opened, and whether the abstract supports the sentence in the paper. Checked 22 October 2026 unless the note says the publisher page was not re-opened.

| Key | Link | Checked |
|---|---|---|
| lewis2020rag | https://arxiv.org/abs/2005.11401 | yes, generation conditioned on retrieved text |
| guu2020realm | https://proceedings.mlr.press/v119/guu20a.html | yes, retriever inside pretraining, ICML pages 3929–3938 |
| izacard2021fid | https://arxiv.org/abs/2007.01282 | yes, generative reader over passages, preprint year 2020 |
| borgeaud2022retro | https://arxiv.org/abs/2112.04426 | yes, RETRO, preprint year 2021, 28 authors |
| izacard2023atlas | https://arxiv.org/abs/2208.03299 | yes, few-shot retrieval-augmented models, preprint year 2022 |
| karpukhin2020dpr | https://arxiv.org/abs/2004.04906 | yes, cited for the Wikipedia passages |
| gao2022tevatron | https://arxiv.org/abs/2203.05765 | yes, toolkit that released those passages |
| yang2018hotpotqa | https://arxiv.org/abs/1809.09600 | yes, multi-hop questions and supporting facts |
| robertson2009bm25 | https://doi.org/10.1561/1500000019 | link only; the journal page was not re-opened this pass |
| kwiatkowski2019nq | https://aclanthology.org/Q19-1026/ | yes, real search queries with long and short answers |
| rajpurkar2016squad | https://arxiv.org/abs/1606.05250 | yes, for exact match and token F1; anthology pages were not opened |
| yao2023react | https://arxiv.org/abs/2210.03629 | yes, reasoning interleaved with actions |
| asai2024selfrag | https://arxiv.org/abs/2310.11511 | yes, retrieve, generate, and critique |
| yan2024crag | https://arxiv.org/abs/2401.15884 | yes, correct a bad retrieval |
| jeong2024adaptive | https://arxiv.org/abs/2403.14403 | yes, a classifier picks retrieval depth from the question |
| jin2025searchr1 | https://arxiv.org/abs/2503.09516 | yes, RL teaches the language model to search |
| gandhi2026grasp | https://arxiv.org/abs/2607.10463 | yes, with the correction below |
| williams1992reinforce | https://doi.org/10.1007/BF00992696 | link only; the journal page was not re-opened this pass |
| qwen2024 | https://arxiv.org/abs/2412.15115 | yes, Qwen2.5 technical report |
| trivedi2023ircot | https://arxiv.org/abs/2212.10509 | yes, retrieval interleaved with chain-of-thought, preprint year 2022 |
| jiang2023flare | https://arxiv.org/abs/2305.06983 | yes, retrieve again when generation is uncertain |
| su2024dragin | https://arxiv.org/abs/2403.10081 | yes, retrieval from the model's information need |
| press2023selfask | https://arxiv.org/abs/2210.03350 | yes, follow-up questions, preprint year 2022 |
| schick2023toolformer | https://arxiv.org/abs/2302.04761 | yes, the model inserts tool calls, including search |
| ma2023rewrite | https://arxiv.org/abs/2305.14283 | yes, rewrite, then retrieve, then read |
| gao2023hyde | https://arxiv.org/abs/2212.10496 | yes, embed a hypothetical passage, preprint year 2022 |
| nogueira2019monobert | https://arxiv.org/abs/1901.04085 | yes, BERT passage reranking |
| sun2023rankgpt | https://arxiv.org/abs/2304.09542 | yes, a language model as a reranker |
| schulman2017ppo | https://arxiv.org/abs/1707.06347 | yes, clipped policy gradient |
| shao2024deepseekmath | https://arxiv.org/abs/2402.03300 | yes, GRPO is the group-relative policy gradient in the abstract |
| song2025r1searcher | https://arxiv.org/abs/2503.05592 | yes, RL incentive for search |
| chen2025research | https://arxiv.org/abs/2503.19470 | yes, RL for reasoning with search |
| chen2023frugalgpt | https://arxiv.org/abs/2305.05176 | yes, cascade across LLM APIs with different prices |
| ong2024routellm | https://arxiv.org/abs/2406.18665 | yes, route each query to a strong or a weak model |
| gao2023rarr | https://arxiv.org/abs/2210.08726 | yes, look up evidence and revise, preprint year 2022 |
| honovich2022true | https://arxiv.org/abs/2204.04991 | yes, compare factual-consistency metrics |

## GRASP

The HTML at https://arxiv.org/abs/2607.10463 gives the title "GRASP: GRanularity-Aware Search Policy for Agentic RAG" and the author Ryan Rossi, not "Ryan A. Rossi." The policy is Qwen2.5-3B-Instruct trained with GRPO. Actions are semantic search, keyword search, and read paragraph. The answer reward is token F1. Turn efficiency is paid only when answer F1 exceeds 0.5, and it counts turns, not dollars. This paper cites GRASP only for that gate.

## What this paper does not claim

FrugalGPT and RouteLLM price which model to call. They do not price retrieve, rewrite, rerank, or verify inside one RAG episode. SQuAD is the source of exact match and token F1 only.
