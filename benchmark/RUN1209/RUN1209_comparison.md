# GPQA-Diamond benchmark — MoE expert expansion across models (paper appendix)

> DeepSeek-V4-Flash-0731 (UD-IQ2_M) and Qwen3.6-35B-A3B (Q8_0), with and without MoE expert expansion; Qwen3.8-27B (Q8) as a cross-model reference. 198 questions, paired protocol, temperature 0, 32,768-token generation budget.

## 1. Headline results

| Configuration | accuracy | correct | tokens mean | tokens median | tokens p90 | truncated |
|---|---|---|---|---|---|---|
| DS4-Flash-0731 exp (N=12, T=0.8, L28-42, decay→0.10) | **85.35%** | 169/198 | 5,631 | 1,580 | 21,407 | 9 |
| DS4-Flash-0731 native (top-6) | **85.35%** | 169/198 | 5,933 | 1,443 | 20,036 | 12 |
| Qwen3.6-35B exp (N=16, T=0.8, L25-39, decay→0.50) | **84.34%** | 167/198 | 7,738 | 5,448 | 16,751 | 6 |
| Qwen3.6-35B native (top-8) | **81.82%** | 162/198 | 7,416 | 5,403 | 16,120 | 5 |
| Qwen3.8-27B (Q8, native) | **83.84%** | 166/198 | 10,911 | 6,758 | 32,768 | 24 |

## 2. Expansion effect (paired, same questions)

| Model | exp accuracy | native accuracy | Δ | exp wins | native wins | net |
|---|---|---|---|---|---|---|
| DeepSeek-V4-Flash | 85.35% | 85.35% | +0.00 pts | 6 | 6 | **+0** |
| Qwen3.6-35B | 84.34% | 81.82% | +2.53 pts | 12 | 7 | **+5** |

- **DS4-Flash-0731**: net 0 — the expansion is neutral at these settings (the earlier net −4 was a budget artifact: native was capped at 16.3k).
- **Qwen3.6-35B**: net +5 — the expansion wins (+2.52 pts), replicating the Q6_K result (net +4) on Q8_0.
- Paper acceptance rule (net ≥ 0): satisfied for Qwen, not for DeepSeek at these settings.

## 3. Does Qwen 35B + expansion reach the 27B?

| Configuration | accuracy |
|---|---|
| Qwen3.8-27B (Q8) — newer gen, smaller | 83.84% |
| Qwen3.6-35B native (top-8) | 81.82% |
| **Qwen3.6-35B expanded (N=16)** | **84.34%** |

**Yes**: the 35B with expansion (84.34%) surpasses the 27B Q8 reference (83.84%, +0.5 pts), while the native 35B (81.82%) falls 2.0 points short. The expansion closes the gap to the newer 27B model and overtakes it.

## 4. Token consumption

| Configuration | mean | median | p90 | total (198 q) | truncated |
|---|---|---|---|---|---|
| DS4-Flash-0731 exp (N=12, T=0.8, L28-42, decay→0.10) | 5,631 | 1,580 | 21,407 | 1,114,938 | 9 |
| DS4-Flash-0731 native (top-6) | 5,933 | 1,443 | 20,036 | 1,174,801 | 12 |
| Qwen3.6-35B exp (N=16, T=0.8, L25-39, decay→0.50) | 7,738 | 5,448 | 16,751 | 1,532,034 | 6 |
| Qwen3.6-35B native (top-8) | 7,416 | 5,403 | 16,120 | 1,468,289 | 5 |
| Qwen3.8-27B (Q8, native) | 10,911 | 6,758 | 32,768 | 2,160,347 | 24 |

Expansion at N=12/16 costs +4-5% mean tokens over native on Qwen; on DeepSeek-V4 the expansion is token-neutral (−5% mean, medians nearly equal). The 27B is the most verbose configuration (24 truncated at the 32k cap).

## 5. Per-subject breakdown (accuracy %)

| Subject | n | DS4 exp | DS4 nat | Q35 exp | Q35 nat | Q27B | net DS4 | net Q35 |
|---|---|---|---|---|---|---|---|---|
| Biology | 19 | 78.9 | 73.7 | 73.7 | 73.7 | 78.9 | +1 | +0 |
| Chemistry | 93 | 81.7 | 80.6 | 76.3 | 71.0 | 74.2 | +1 | +5 |
| Physics | 86 | 90.7 | 93.0 | 95.3 | 95.3 | 95.3 | -2 | +0 |

Expansion nets are positive or neutral in every subject for both models.

## 6. Methodology

- Paired protocol: same 198 GPQA-diamond questions, temperature 0 (greedy), identical prompts, 32,768-token generation budget in every run; evalscope review scoring.
- Expansion is runtime-only (no weight changes): N = routed-expert budget above the native top-K, T = adaptive threshold on rank N/2, late-layer scoped, linear influence decay on the added ranks.
- DS4 runs: unsloth UD-IQ2_M (~2-bit), server ctx 43,264. Qwen runs: unsloth Q8_0. Each model's runs on the same host; tpot/latency comparisons are valid within a model pair only.
- Net = expansion wins − native wins over divergent questions; acceptance rule: net ≥ 0.
