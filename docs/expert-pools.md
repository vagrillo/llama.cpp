# Per-Expert Mixed-Precision Quantization (split pools)

Per-expert mixed-precision quantization for MoE models, driven by runtime
expert-usage ranking: the most-used experts of every layer are quantized at
higher precision, the tail at lower precision.

Reference model published on HuggingFace:
**[vagrillo/Qwen35B-A4B-MOExpMixedQuant](https://huggingface.co/vagrillo/Qwen35B-A4B-MOExpMixedQuant)**
(Qwen3.6-35B-A3B, 4 pools: top-16 Q8_0 / 17-64 Q6_K / 65-192 Q4_K / 193-256
Q2_K per layer, shared expert at Q8_0 — ~21 GB vs 37 GB for the uniform Q8_0).

## Components

### 1. Quantization with split (`llama-quantize --split-experts`)

```
llama-quantize --allow-requantize \
  --imatrix imatrix_unsloth.gguf \
  --split-experts expert-order.txt \
  Qwen3.6-35B-A3B-Q8_0.gguf Qwen3.6-35B-A3B-mixed-expert.gguf Q8_0
```

`expert-order.txt`: one line per MoE layer, 256 comma-separated expert ids
sorted by usage rank (descending). For each routed-expert tensor of the source
(`ffn_gate_up_exps`, or `ffn_gate_exps` + `ffn_up_exps` when the source uses
separate gate/up, and `ffn_down_exps`) it emits **4 pool tensors**
(`_hi`/`_mid`/`_lo`/`_tail`, 16/48/128/64 experts) quantized with the tier
type, plus **8 F32 [n_expert] tables** per layer (`ffn_exps_local_0..3`,
`ffn_exps_mask_0..3`): `local_p[e]` = local index of expert `e` in pool `p`
(0 if out of pool), `mask_p[e]` = 1.0 if in pool. The imatrix is per-expert
(`[n_cols, n_expert]`) and is sliced along the expert axis so every pool is
quantized with its own imatrix portion. All other tensors (attention,
embeddings, output, shared expert, router, norms) go through the standard path.

### 2. Graph with pools (`build_moe_ffn`, pool mode)

The `qwen35moe` loader detects the pool tensors; `build_moe_ffn` receives the 4
pools and the tables and, with the routing unchanged (same softmax, same
top-20, same expansion and threshold), for each pool:

- remaps router ids to the local index: `ids_p = cast_i32(get_rows(local_p, ids))`
- masks the weights: `w_p = weights * get_rows(mask_p, ids)` (out-of-pool rows
  evaluate local expert 0 with weight 0: zero contribution, static shapes)
- computes the pool FFN (`mul_mat_id` fused gate_up or separate gate/up →
  swiglu → down)
- sums the partials of all pools

The math is identical to the non-pooled compute; only each expert's precision
differs. The expert-usage tracking (`LLAMA_MOE_EXPERT_USAGE_FILE`, below)
keeps working because the routing is untouched.

Performance note: the per-pool compute evaluates `n_used` rows per pool
(4 × 20 expert evaluations per token instead of 20) — the FFN overhead is ~4×
in decode. **Per-pool budgets** remove most of it: with
`LLAMA_MOE_POOL_BUDGETS=K0,K1,K2,K3` every pool sorts its slots by masked
weight per token (argsort desc) and evaluates only the first `K_p` rows; the
dropped slots carry negligible or zero weight. Suggested budgets from the
usage ranking: `10,5,3,2` (20 rows total instead of 80) — measured on Apple
Silicon: ~2× prefill and a large decode speedup; `K_p ≥ n_used` or unset env
keeps the dense behavior. The compacted ids may contain duplicate local
experts (zero-weight slots), which every `mul_mat_id` path compacts per
(token, slot) pair since the per-pair fixes.

## Expert-usage tracking (to generate the ranking)

`LLAMA_MOE_EXPERT_USAGE_FILE=path.json` enables the cumulative per-(layer,
expert) count of all kept experts (post-expansion weight > 1e-6):

```bash
LLAMA_MOE_EXPERT_USAGE_FILE=usage.json LLAMA_MOE_EXPERT_USAGE_EVERY=256 \
  llama-server -m model.gguf --moe-experts 20 --moe-expert-threshold 0.95 ...
```

The JSON is atomically rewritten every `EVERY` tokens:

```json
{"format": "moe-expert-usage-v2", "total_tokens": 441745,
 "layer_tokens": {"0": 441745, ...},
 "layers": {"0": {"n_expert": 256, "counts": [1203, 87, ...]}, ...}}
```

`counts[E]` = how many times expert `E` was kept by the router over all counted
tokens. The readback happens in the **sched eval callback** (per-tensor sync,
`moe_chunk_cb` pattern): the end-of-graph readback loses decode ubatches to
buffer aliasing (CUDA graphs + concurrent topologies). An external poller
accumulates the deltas into a persistent cross-run total. Logging note:
`LLAMA_LOG_INFO` is filtered by default in this fork — dump events do not
appear in the standard logs.

Also required for multi-token batches: the `mm_ids_helper` CUDA kernel now
compacts **per (row, token) pair** — with per-expert pools, several rows of the
same token map to the same local expert and the per-distinct-expert
compaction left uninitialized `ids_src1` entries (illegal memory access in the
MMQ scatter). Standard models (all pairs distinct) are unaffected.

The same duplicate-id assumption broke the CUDA `mul_mat_id` **sorted
fallback** (reached on tensor-core GPUs such as V100 for quantized weights
when the token batch exceeds `MMQ_DP4A_MAX_BATCH_SIZE`): its host loop kept
only the first slot per (token, expert) pair, under-counting the compact rows
until `GGML_ASSERT(ids_to_sorted_host.size() == ne_get_rows)` aborted the
server on the first large prefill. The fallback now compacts per pair as
well, so every (token, slot) pair gets its own row on every path. The pool
debug prints (`POOLDBG*`) are gated behind `LLAMA_MOE_POOL_DEBUG=1`.

## Reference workload and results

Ranking collected on GPQA-Diamond (93 questions, greedy replay, reasoning
budget 8192, concurrency 4) on V100-32GB: 441,745 tokens, 20,591,738
selections, all 40 layers validated at ~11.5 selections/token. Method, caveats
(domain bias, nearly flat tail) and files:
[vagrillo/AgrillaMoE](https://github.com/vagrillo/AgrillaMoE), `expert-rank/`
directory (`EXPERT-RANK.md`).
