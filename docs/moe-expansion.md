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

**Universal across MoE families**: it works on any architecture routed through
the standard MoE path — Qwen MoE, DeepSeek (V4), GLM 4/5.x, LiquidAI LFM2-MoE,
Gemma 4, gpt-oss, OLMOE, Mixtral, Jamba, MiniMax, Kimi, Nemotron-H and many
more (see the coverage table below).

Reference results on Qwen3.6-35B-A3B (ds4/Metal, M4 Pro, MMLU-Pro 714 questions,
greedy): native top-8 vs expansion N=20 T=0.8 decay 0.99→0.50 — accuracy 84.0%
vs 84.5% (unchanged), mean reasoning tokens −8.5%, latency −10.9%, with ~15.5
experts/token instead of 8. Gains are model-dependent: always measure per model.
On GPQA-Diamond the accuracy effect is where the expansion stands out: see the
RUN1209 results below (+2.52 pts on Qwen3.6-35B-A3B, paired net +5).

## Measured on GPQA-Diamond (RUN1209, this branch)

Five-run paired benchmark — 198/198 GPQA-Diamond questions per run, greedy,
identical prompts, generation cap 32,768 tokens for every run; the only
changed variable is the routing configuration:

| run | model | quant | routing | accuracy |
|---|---|---|---|---|
| DeepSeek-V4-Flash-0731 | expanded | UD-IQ2_M (~2-bit) | N=12, T=0.8, L28–42, decay→0.10 | **85.35%** |
| DeepSeek-V4-Flash-0731 | native | UD-IQ2_M (~2-bit) | top-6 | **85.35%** |
| Qwen3.6-35B-A3B | **expanded** | Q8_0 | N=16, T=0.8, L25–39, decay→0.50 | **84.34%** |
| Qwen3.6-35B-A3B | native | Q8_0 | top-8 | 81.82% |
| Qwen3.8-27B (dense, reference) | native | Q8_0 | — | 83.84% |

> **The accuracy advantage is notable.** On the same Q8_0 checkpoint the
> expansion gains **+2.52 pts** over native routing (paired net **+5**:
> 12 wins vs 7 losses, agreement on 179/198 questions). With the expansion
> active, the **Qwen3.6-35B-A3B (Q8) overtakes the newer dense Qwen3.8-27B**
> (84.34% vs 83.84%): native routing falls 2.0 pts short of the 27B — the
> expansion closes the generation gap. The gain is concentrated in Chemistry
> (70.97% → 76.34%; 10 of the 12 paired wins), with Physics saturated at
> 95.35% for the whole family. Token cost: +4.3% mean over the full run,
> **+9.5% restricted to the matched both-correct questions** (medians flat —
> the gap sits in the long tail).

On DeepSeek-V4-Flash-0731 the expansion is neutral at 2-bit (paired net 0,
identical 85.35%, −5% mean tokens, 3 fewer truncations): consistent with the
model-dependence warning — at IQ2_M the router scores beyond the native top-6
carry more noise than signal. The cost of wider routing is ~11–13% generation
throughput on both families; full performance analysis (latency, tok/s,
GPU-hours) in the final report.

Complete final report: [benchmark/RUN1209/finalcompariso1209.md](../benchmark/RUN1209/finalcompariso1209.md)
([HTML rendered](https://htmlpreview.github.io/?https://github.com/vagrillo/llama.cpp/blob/moe-expansion/benchmark/RUN1209/finalcompariso1209.html),
[source](../benchmark/RUN1209/finalcompariso1209.html) — GitHub does not render
HTML; download and open locally) — raw predictions, reviews, per-subject and
token-class breakdowns, loop analysis and methodology are committed under
[benchmark/RUN1209/](../benchmark/RUN1209/).

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
| `--moe-expert-renorm MODE` | — | `auto` | kept-weight renormalization after cut+decay: `auto` follows the model's stock normalization (renorm iff `expert_weights_norm=true`), `always` forces sum-to-1, `never` keeps the raw decayed score scale |
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

### Model coverage

The feature lives in the shared `llm_graph_context::build_moe_ffn` router, so
it applies to every MoE architecture that uses the standard routing path —
regardless of the architecture name. Supported router families:

| router family | models (examples) | notes |
|---|---|---|
| softmax | qwen2/3/3.5 MoE, qwen3vlmoe, qwen4exp, glm4-moe, lfm2moe (LiquidAI), olmoe, mixtral, smallthinker, jamba, minimax, nemotron-h, kimi, bailing, cohere2moe, ernie, llada-moe, hunyuan, grok, rnd1, mimo2, mellum, laguna, dots3note, dflash, ... | reference family (Qwen3.6-35B-A3B) |
| softmax + selection bias | deepseek2, deepseek32, deepseek4 (V4), glm-dsa (GLM 5.x) | bias affects rank order only; the cut is self-consistent with the gathered weights |
| sigmoid + weight norm | glm-dsa (GLM 5.x default), ... | scores renormalized over the kept set, same as stock |
| softmax-of-selected-scores | openai-moe (gpt-oss) | softmax over the selected set runs before the post-pass |
| sqrt-softplus | deepseek4 (V4) | positive unnormalized scores; the post-pass renormalizes (spec §2) |
| precomputed router logits | gemma4 | `probs_in` is fine: the gating softmax normalizes it |

Not supported (native routing kept, no error):

- grouped expert routing (`n_expert_groups > 1`, DeepSeek V3-style group
  top-k) — requesting the flags on such a model fails fast at context creation
- models that apply the expert weights **before** the FFN without
  normalization (llama4) — the renormalization would change stock semantics
- custom expert selections (`selected_experts_in`) and MTP/draft graphs

### Mechanics

- The selection/cut/decay/renormalization is implemented as pure graph ops on
  the rank-ordered router weights in `llm_graph_context::build_moe_ffn`
  (`src/llama-moe-expansion.h`), so it runs identically on every backend
  (CPU, Metal, CUDA, Vulkan, ...). Strategy: **fixed-N mask** — the graph
  always carries N slots per token and dropped ranks carry zero weight, so the
  stock expert kernels need no changes. Consequence: compute scales with N (not
  with the adaptive count), i.e. speed ~N/K of native. `T > 1` pruning
  therefore saves quality-affecting compute only on engines with true
  variable-count execution (ds4 does); here it is a routing change, not a
  speedup.
- Weight renormalization is mode-controlled (`--moe-expert-renorm`, default
  `auto`): for softmax routers with stock normalization (Qwen3.6: norm + routed
  scale) the kept weights are renormalized to sum 1 as in stock; for raw-score
  routers (DeepSeek-V4's sqrt-softplus + bias with `expert_weights_norm=false`)
  `auto` does NOT renormalize — the kept weights keep their decayed score scale
  and the dropped mass is discarded, exactly like the ds4 DeepSeek-V4 path.
  Forcing `never`/`always` overrides this.
  <br><br>Measured on GPQA-Diamond (DeepSeek-V4-Flash-0731, IQ2_M, RUN1209):
  with `auto` renormalization at equal 32k budgets the expansion is **neutral**
  (85.35% native vs 85.35% expanded, paired net 0, −5% mean tokens) — the
  earlier regression (−2.02 pts, paired net −4) had been produced by a
  pre-`auto` build that forced renormalization, compounded by an unequal-budget
  comparison (native capped at 16.3k). The renormalization mode exists
  precisely to control the weight-sum semantics on raw-score routers:
  under `auto` DeepSeek-V4 keeps its raw decayed scores (no renormalization),
  exactly like the ds4 DeepSeek-V4 path.
- Warmup batches keep the native routing (the warmup graph already exercises
  all experts, and it sizes the compute buffers: N ≤ expert_count fits).
- Expert-id tie-breaking on equal router probabilities follows ggml's argsort
  (not ds4's lowest-id scan); ties in float probabilities are a negligible
  edge case.

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

## Building and running on macOS (Metal)

All expansion operators (FILL, ARANGE, STEP, CLAMP, REPEAT, SUM_ROWS,
ARGSORT, GET_ROWS, broadcast arithmetic) have standard Metal kernels in this
repository — no extra kernels are needed for the expansion on Apple Silicon.

The classic macOS build failure is a **missing Metal toolchain at build
time**: CMake used to skip the `default.metallib` generation with only a
warning, producing a binary that reports *"can't find kernel"* errors and
often crashes at the first GPU compute. The build now **fails with a clear
error** when `GGML_METAL=ON` and `xcrun metal` is not available.

Fix and verification:

```bash
xcode-select --install                       # Metal toolchain comes with Xcode/CLT
xcrun -f metal && xcrun -f metallib          # both must print a path

rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL_EMBED=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

`-DGGML_METAL_EMBED=ON` compiles the kernels into the binary itself: no
runtime lookup of `default.metallib`, immune to "running from another
directory" issues. Recommended for shared builds.

Alternatives:

- `-DGGML_METAL=OFF` — CPU-only build; useful to isolate whether a problem is
  the Metal build or the code (if CPU runs and Metal crashes, it is the build).
- If a segfault happens anyway with a working Metal build, capture a backtrace:

```bash
lldb -- ./build/bin/llama-server -m <model> --moe-experts 12 --moe-expert-threshold 0.8 ...
(lldb) run
(lldb) bt
```

and open an issue with the backtrace plus the `moe:` startup banner lines.

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
- **Quality benchmark**: RUN1209 — five complete paired GPQA-Diamond runs
  (198/198 questions each, greedy, equal 32k budgets) across two model
  families and a dense reference: expansion **+2.52 pts (paired net +5)** on
  Qwen3.6-35B-A3B Q8_0 — overtaking the dense Qwen3.8-27B reference — and
  **neutral (net 0)** on DeepSeek-V4-Flash-0731 IQ2_M. Full data, methodology,
  per-subject/token-class/loop analysis and final report:
  [benchmark/RUN1209/](../benchmark/RUN1209/) ·
  [final report (md)](../benchmark/RUN1209/finalcompariso1209.md) ·
  [final report (html, rendered)](https://htmlpreview.github.io/?https://github.com/vagrillo/llama.cpp/blob/moe-expansion/benchmark/RUN1209/finalcompariso1209.html).

When benchmarking quality, keep the paired protocol (same questions, greedy,
one variable at a time) — see the ds4 instruction document for the full
MMLU-Pro recipe, and
[benchmark/RUN1209/finalcompariso1209.md](../benchmark/RUN1209/finalcompariso1209.md)
for the GPQA-Diamond protocol used by RUN1209 (paired exp/native runs, equal
token budgets, evalscope `ANSWER: [LETTER]` scoring).
