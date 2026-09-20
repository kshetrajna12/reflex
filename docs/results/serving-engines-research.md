# Prior art: the Jev readout on production inference engines (research note, 2026-09-20)

Question: has anyone put the shared-state, per-question logit readout on vLLM or SGLang, so
reflex can reuse it instead of its transformers-only engine?

**Yes, on SGLang.** [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang) is a
Jev-API-compatible server: it renders the chat prefix once, warms SGLang's radix cache by
sending the prefix with `max_new_tokens=1`, then fires one concurrent `/generate` per question
(`prefix + question + assistant header`) with `max_new_tokens=1`, `return_logprob=true`,
`logprob_start_len=-1` and `token_ids_logprob=[label ids]`, and softmaxes the returned exact
log-probabilities. Isolation between questions comes from separate requests; no attention
mask, no copied cache. Running Qwen3.6-35B-A3B in NVFP4 on SGLang 0.5.19 it scores 95.5 % on
JevBench v1.2 against Jev 1.13.0's 96.3 % (overlapping 95 % CIs).

**Not on vLLM, yet.** Exact per-token-id log-probabilities are not in a released vLLM: the
`track_token_ids` request (issue 29280) was closed as not planned; `prompt_logprob_token_ids`
(RFC 56860, PR 54335) is open, V2-runner only, with no OpenAI-server field. Projects on vLLM
([ikermoel/open-alternative-jev](https://github.com/ikermoel/open-alternative-jev)) use top-k
`prompt_logprobs` (k = 20) with a floor for labels outside the top-k, which is lossy. vLLM's
`classify` / `score` / `pooling` APIs need a classification head and do not apply. And vLLM's
prefix cache for attention+Mamba hybrids only caches full 528-token blocks (issue 40696: 0 %
hit rate below the boundary), so short states get no reuse on Qwen3.5/3.6/3.8. SGLang's
unified radix cache handles the hybrids without that cliff.

Other entries on the JevBench board ([SemIf](https://github.com/TheoLeeCJ/SemIf),
localjev, the encoder-based classifiers) use custom transformers or MLX backends or are not
logit readouts at all. The Archer Hume article infers "prefix cache plus separate causal
suffixes" from ~10,000 timed API calls and disclaims knowing TypeSafe's stack; TypeSafe has
disclosed nothing.

**Consequence for reflex:** an SGLang backend following the openjev-sglang recipe, behind the
same `/v1/systemone` API and prompt, gated by the transformers-equivalence test and the
external sets. Two SGLang caveats to code around: mixed batches where some requests omit
`token_ids_logprob` (issue 30188), and `token_ids_logprob` applying the union of candidate
ids at every position. Request raw log-probabilities (`logprobs_mode`), never post-sampling.
