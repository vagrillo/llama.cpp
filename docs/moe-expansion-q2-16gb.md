# Qwen3.6-35B-A3B at 2-bit on a 16 GB GPU

**Goal:** run the 35B MoE on a **16 GB** consumer GPU. The Q8_0 build needs
**36.9 GB** of VRAM for the weights alone (a 40–48 GB card); the
**UD-Q2_K_XL** build needs **12.29 GB** — it fits a 16 GB card with room for
context. On its own, a 2-bit quantization of this family loses accuracy on
hard reasoning; combined with **moe-expansion** and the **reasoning budget**,
the 2-bit model reaches **the same GPQA-Diamond accuracy as the Q8_0 native
build** (81.82%, paired net 0) at roughly **one third of the VRAM**.

Feature reference: [docs/moe-expansion.md](moe-expansion.md) ·
benchmark data: [benchmark/RUN1209/](../benchmark/RUN1209/)

---

## Recommended configuration (validated)

```bash
./llama-server \
  -m Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf \
  -ngl 999 -c 142768 --temp 0 --jinja --parallel 4 \
  --moe-experts 20 --moe-expert-threshold 0.8 \
  --moe-expert-decay-end 0.5 --moe-expert-layer-start 25 \
  --reasoning-budget 22480 \
  --reasoning-budget-message "Stop reasoning now: commit to your best-supported answer and respond with the final answer." \
  --host 0.0.0.0 --port 9080
```

| parameter | value | why |
|---|---|---|
| `--moe-experts 20` | expansion cap | raises the routed budget above the native top-8 (adaptive 5–20 per token) |
| `--moe-expert-threshold 0.8` | adaptive | peaked tokens route fewer experts, dense tokens more; keeps the token cost bounded |
| `--moe-expert-decay-end 0.5` | influence fade | extra ranks fade 0.99→0.50; native experts keep the output scale |
| `--moe-expert-layer-start 25` | late layers | layer-scoped expansion (layers 25–39 of 40), where the budget matters most |
| `--reasoning-budget 22480` | **required at 2-bit** | forces end-of-thinking at ~22k tokens and makes the model commit to an answer; without it, low-bit chains can drift into repetition and burn the whole token cap |
| `--reasoning-budget-message "Stop reasoning now: …"` | wrap-up prompt | injected before the end-of-thinking tag so the forced answer is a committed one |
| `--temp 0` | greedy | deterministic, benchmark-grade decoding |
| `-c 142768 --parallel 4` | context/slots | validated configuration; see the 16 GB notes below |

Keep the client `max_tokens` comfortably above the reasoning budget
(32,768 works: ~22.5k reasoning + ~10k for the final answer).

### 16 GB-specific notes

The benchmark configuration above was validated on a 32 GB datacenter GPU
(V100-class) with the full 142k context across 4 slots. On a 16 GB card the
weights (12.29 GB) leave roughly 3–3.5 GB for context and compute buffers:

- start at `-c 20480` (single slot) and raise while VRAM allows;
- `--kv-cache-type q8_0` roughly doubles the affordable context at a negligible quality cost;
- keep `--parallel 1`–`2`; each extra slot multiplies the KV requirement;
- everything else (expansion knobs, reasoning budget) stays identical — they
  do not add VRAM.

## Measured quality (GPQA-Diamond, 198 questions, greedy)

| configuration | weights | accuracy | paired vs Q8 native |
|---|---|---:|---|
| Q8_0 native (top-8) | 36.9 GB | 81.82% | — |
| **UD-Q2_K_XL + expansion + budget** | **12.3 GB** | **81.82%** | **net 0 — the same 162 questions solved** |
| Q8_0 + expansion | 36.9 GB | 84.34% | +2.52 (reference ceiling) |

Per subject (Q2 + expansion + budget): Chemistry 72.0% · Physics 95.4% ·
Biology 68.4%. The residual −2.53 vs the Q8_0 expanded build is the true
quantization cost at this bit-rate; against the *native* Q8_0 build — the
fair baseline for a fixed 16 GB budget — the 2-bit pipeline is at parity.

Throughput on the validation GPU (V100-class, batch 4): **42 tok/s per
request**, ~23 ms/token, mean latency 182 s, p90 output 22.9k tokens
(under the budget). Protocol: paired, same 198 questions and prompt,
greedy, 32,768-token cap, equal budgets across runs
([benchmark/RUN1209/](../benchmark/RUN1209/) methodology).

## Why the reasoning budget is part of the recipe

At 2-bit, long reasoning chains on hard questions can drift into repetition
loops: the model cycles a hypothesis until the token cap is exhausted and
returns no answer at all. `--reasoning-budget` closes the thinking block
while the model still has budget for a final answer, and the injected
message pushes a committed choice. In the validated run this kept the
output tail under the budget (p90 22.9k), removed cap-truncated responses
entirely, cut total tokens by ~13% at unchanged throughput — and is the
difference between the parity result above and a materially lower score.

**Recommendation: keep `--reasoning-budget` enabled for any sub-Q4
deployment of this model family.** The parity result (2-bit + expansion +
budget = Q8_0 native accuracy at ~1/3 VRAM) is what makes the 16 GB target
possible.
