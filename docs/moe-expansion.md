# MoE Expert Expansion

Runtime-only routing modification for sparse MoE models: raise the routed-expert
budget above the model's native top-K with a linear influence decay on the extra
ranks, optionally with an adaptive (threshold-based) expert count and a layer
range. Universal: it works on any MoE architecture that routes through the
standard softmax top-k path (qwen2moe / qwen3moe / qwen35moe, deepseek*, glm,
hunyuan, and many more).

This is the llama.cpp port of the "layer-scoped expert-budget expansion" feature
of DwarfStar (ds4, branch `qwen35moe-support`), described in
[Vincenzo Agrillo, *Succinct convergence...* (Zenodo, 2026)](https://zenodo.org/records/22255483).
No model-file changes, no GGUF changes, no kernel changes: with the feature off,
the engine behaves bit-identically to stock.

Reference results on Qwen3.6-35B-A3B (ds4/Metal, M4 Pro, MMLU-Pro 714 questions,
greedy): native top-8 vs expansion N=20 T=0.8 decay 0.99→0.50 — accuracy 84.0%
vs 84.5% (unchanged), mean reasoning tokens −8.5%, latency −10.9%, with ~15.5
experts/token instead of 8. Gains are model-dependent: always measure per model.

## The three knobs

1. **Max experts N** (`--moe-experts N`, or `--moe-experts-add N` to add on top
   of the native K): raise the routed expert budget above the model's native
   top-K (e.g. 8 → 20). Values below the native K are also allowed (fixed
   pruning).
2. **Dynamic threshold T** (`--moe-expert-threshold T`): instead of a fixed
   count, keep experts while their router probability stays above `T × p_ref`,
   where `p_ref` is the probability of rank `N/2`. The per-token expert count
   becomes adaptive: dense tokens route to ~N experts, peaked tokens to fewer.
   `T ≤ 1` only ever expands past the native K; `T > 1` can also prune below it,
   down to the floor `max(1, N/4)`.
3. **Expansion decay D** (`--moe-expert-decay-end D`, default 0.50): when
   N > K, ranks above the native K get a linearly decaying influence factor
   from 0.99 down to D (single extra rank gets the midpoint 0.745), applied to
   the raw probability *before* renormalization, so the native experts keep the
   output scale and the extras fade out. `--moe-no-expert-decay` disables the
   decay (extra experts at full influence).

All flags off → bit-identical to stock (hard requirement, verified below).
`N == K` with no threshold is also exactly stock.

## CLI flags

| flag | alias (ds4) | default | meaning |
|---|---|---|---|
| `--moe-experts N` | `--q35-experts` | 0 (off) | absolute max routed experts per token N (2 ≤ N ≤ expert_count) |
| `--moe-experts-add N` | — | 0 (off) | experts added on top of the model's native top-K |
| `--moe-expert-threshold T` | `--q35-expert-threshold` | 0 (off) | adaptive count: keep while p ≥ T × p(rank N/2); T ∈ (0, 10] |
| `--moe-expert-decay-end D` | — | 0.50 | influence of the last extra rank; linear 0.99..D; D ∈ (0, 0.99) |
| `--moe-no-expert-decay` | `--q35-no-expert-decay` | off | extra experts at full influence |
| `--moe-expert-layer-start I` | — | 0 | first layer the expansion applies to; `< 1`: fraction of n_layer, `≥ 1`: layer index |
| `--moe-expert-layer-end I` | — | −1 | last layer (inclusive); `< 0`: last layer, `< 1`: fraction, `≥ 1`: index |

The layer range is what makes the technique "layer-scoped" (per the paper, the
budget expansion matters most in late layers): e.g. `--moe-expert-layer-start
0.5` restricts expansion to the second half of the layers.

Invalid parameters (N out of range, non-MoE model, grouped expert routing,
threshold outside (0, 10], decay end outside (0, 0.99), layer start > end) fail
fast at context creation with a clear message and a non-zero exit code.

## Observability

With the feature active, every `LLAMA_MOE_EXPERT_STATS_EVERY` processed tokens
(default 128, `0` = off) the engine prints to stderr the average number of
routed experts per token, per expanded layer:

```
moe: routed experts per token: 20 (model default 8; ranks 9..20 get linear influence decay 0.99..0.50)
moe: expert threshold: keep experts while p >= 0.80 x p(rank 10) (experts 5..20 per token)
moe: expansion active on layers 20..39 of 40
...
moe: experts/token avg over 128 tokens: L20:16.1 L21:15.4 ... | mean 15.5
```

This is the primary health signal: with `N=20, T=0.8` on Qwen3.6-35B-A3B expect
a mean around 15-16; exactly 20.0 with `T=0`; exactly `N/4` with `T=10`.

## Implementation notes (llama.cpp specifics)

- The selection/cut/decay/renormalization is implemented as pure graph ops on
  the rank-ordered router weights in `llm_graph_context::build_moe_ffn`
  (`src/llama-moe-expansion.h`), so it runs identically on every backend
  (CPU, Metal, CUDA, Vulkan, ...). Strategy: **fixed-N mask** — the graph
  always carries N selection slots per token and dropped ranks carry zero
  weight, so the stock expert kernels need no changes. Consequence: compute
  scales with N (not with the adaptive count), i.e. speed ~N/K of native.
  `T > 1` pruning therefore saves quality-affecting compute only on engines
  with true variable-count execution (ds4 does); here it is a routing change,
  not a speedup.
- Applies only to softmax-router MoE graphs; MTP/draft graphs keep the native
  routing; grouped expert routing (n_expert_groups > 1) and custom selections
  are excluded.
- Warmup batches keep the native routing (the warmup graph already exercises
  all experts, and it sizes the compute buffers: N ≤ expert_count fits).
- Expert-id tie-breaking on equal router probabilities follows ggml's argsort
  (not ds4's lowest-id scan); ties in float softmax probabilities are a
  negligible edge case.
- DeepSeek-style expert-selection bias affects the rank order, as in stock;
  the threshold is self-consistent with the gathered (unbiased) weights.

## Examples

```bash
# quality-leaning expansion, as measured in the paper (Qwen3.6-35B-A3B):
llama-server -m Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf --temp 0 \
    --moe-experts 20 --moe-expert-threshold 0.8          # ~15.5 experts/token

# same, ds4-compatible spelling:
llama-cli -m model.gguf --q35-experts 20 --q35-expert-threshold 0.8

# expansion restricted to the late layers (layer-scoped), decay fading to 0.30:
llama-cli -m model.gguf --moe-experts-add 12 --moe-expert-decay-end 0.30 \
    --moe-expert-layer-start 0.5

# aggressive pruning below the native top-K (c ∈ [2..8] on an 8-expert model):
llama-cli -m model.gguf --moe-experts 8 --moe-expert-threshold 2

# tune/verify with the stats line:
LLAMA_MOE_EXPERT_STATS_EVERY=64 llama-cli -m model.gguf --moe-experts 20 ...
```

## Verification performed on this branch

- `tests/test-moe-expansion.cpp` (22 checks): the graph post-pass matches a
  direct implementation of the normative spec on all golden cases — T=0,
  T=0.8 power-law (c ∈ [R, N]), T=10 → floor, flat probabilities → N, single
  extra → factor 0.745, D=0.30 endpoints, no-decay, per-token independence,
  pruning below K, and a 32-configuration sweep; kept weights sum to 1.
- End-to-end smoke tests on a toy qwen3moe GGUF (`llama-cli` / `llama-server`,
  greedy): flags-off and N == K byte-identical to stock; banner and stats
  lines correct; `T=0` → mean = N exactly; `T=10` → mean = floor; ds4 aliases
  equivalent; deterministic across runs; layer range gates per layer; invalid
  parameters exit non-zero.
- `test-arg-parser`, `test-sampling`, `test-chat-template` pass unmodified.

When benchmarking quality, keep the paired protocol (same questions, greedy,
one variable at a time) — see the ds4 instruction document for the full
MMLU-Pro recipe.
