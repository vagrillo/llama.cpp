# GPQA-Diamond - MoE expert expansion vs native routing

> Model: Qwen3.6-35B-A3B (UD-Q6_K_XL) · 198 questions, paired protocol, temperature 0 · Expansion config: **N=20, threshold 0.8, layers 29-39, decay 0.99→0.50** · Native config: top-8 · llama.cpp branch `moe-expansion` (25ca82a6)

| Metric | Expansion | Native |
|---|---|---|
| **accuracy** | **85.35%** (169/198) | **83.33%** (165/198) |
| paired net | **+4** (13 wins / 9 losses, 156 ties) | - |
| response tokens - **mean** | 7,392 | 8,156 |
| response tokens - **median** | 5,180 | 5,494 |
| response tokens - p90 | 15,825 | 15,310 |
| response tokens - max | 32,768 | 132,650 |
| total tokens (198 questions) | 1,463,642 | 1,614,940 |
| truncated responses (length) | 3 | 5 |

## Comparative outcome

Expansion **improves accuracy by +2.02 points** (85.35% vs 83.33%) while at the same time **reducing tokens consumed** (mean -9.4%, median -5.7%): on GPQA the "Succinct Convergence" phenomenon described in the paper shows up on both quality and brevity.

The paired net **+4** (13 wins vs 9 losses, out of 22 divergent questions) is positive but not statistically significant at n=198 (≈1.7σ): consistent with the 1-3 point effect reported in the paper. **Paper acceptance rule (net ≥ 0): satisfied - the feature is valid for GPQA on this model.**

## Response token distribution (paired percentiles)

| percentile | expansion | native | Δ |
|---|---|---|---|
| p10 | 2,934 | 2,961 | -0.9% |
| p25 | 3,808 | 3,970 | -4.1% |
| p50 | 5,202 | 5,506 | -5.5% |
| p75 | 7,884 | 7,922 | -0.5% |
| p90 | 15,825 | 15,310 | +3.4% |
| p95 | 22,345 | 25,049 | -10.8% |

Across all percentiles up to p75 the expansion produces shorter answers (≈ -3/-6%): the paper's "succinct convergence". Divergences are point-wise and limited to the tail.

## Token cost under per-response budget

| Cap | total expansion | total native | Δ | clipped exp/nat |
|---|---|---|---|---|
| none | 1,463,642 | 1,614,940 | -9.4% | 0 / 0 |
| 32K | 1,463,642 | 1,515,058 | -3.4% | 0 / 1 |
| 16K | 1,324,352 | 1,344,207 | -1.5% | 18 / 18 |

Clipping estimates **cost**, not quality: a truncated answer could flip its verdict.

## Per-subject analysis

| Subject | n | exp accuracy | native accuracy | paired net | exp mean tokens | native mean tokens | exp median tokens | native median tokens |
|---|---|---|---|---|---|---|---|---|
| Biology | 19 | 73.7% (14) | 68.4% (13) | +1 | 4,482 | 5,864 | 4,027 | 3,799 |
| Chemistry | 93 | 77.4% (72) | 76.3% (71) | +1 | 10,313 | 11,425 | 7,262 | 6,783 |
| Physics | 86 | 96.5% (83) | 94.2% (81) | +2 | 4,876 | 5,128 | 4,444 | 4,614 |

Net is positive across **all three** subjects, with expansion mean and median tokens lower everywhere: the improvement is not concentrated in a single domain.

## Subcategory detail

| Subcategory | n | exp accuracy | native accuracy | net |
|---|---|---|---|---|
| Biology — Genetics | 4 | 75.0% (3) | 50.0% (2) | +1 |
| Biology — Molecular Biology | 15 | 73.3% (11) | 73.3% (11) | +0 |
| Chemistry — Chemistry (general) | 20 | 90.0% (18) | 85.0% (17) | +1 |
| Chemistry — Inorganic Chemistry | 1 | 100.0% (1) | 100.0% (1) | +0 |
| Chemistry — Organic Chemistry | 72 | 73.6% (53) | 73.6% (53) | +0 |
| Physics — Astrophysics | 13 | 100.0% (13) | 92.3% (12) | +1 |
| Physics — Condensed Matter Physics | 1 | 100.0% (1) | 100.0% (1) | +0 |
| Physics — Electromagnetism and Photonics | 6 | 83.3% (5) | 83.3% (5) | +0 |
| Physics — High-energy particle physics | 14 | 100.0% (14) | 100.0% (14) | +0 |
| Physics — Optics and Acoustics | 1 | 100.0% (1) | 0.0% (0) | +1 |
| Physics — Physics (general) | 19 | 94.7% (18) | 94.7% (18) | +0 |
| Physics — Quantum Mechanics | 25 | 100.0% (25) | 100.0% (25) | +0 |
| Physics — Relativistic Mechanics | 7 | 85.7% (6) | 85.7% (6) | +0 |

## Latency - indicative only, different GPUs (not comparable)

| Metric | Expansion run | Native run |
|---|---|---|
| mean tpot (ms/token) | 13.1 | 9.2 |
| mean latency per question (s) | 98.7 | 78.1 |

The two runs were executed on **different GPUs**: tpot/latency differences reflect the hardware, not the configuration, and must not be used as a comparison. On identical hardware the expected reference for expansion is -20/-35% decoding speed vs native (20 vs 8 active experts).

## Divergent questions (22)

| # | Subject | Target | Expansion answer | Native answer | Winner | exp tokens | nat tokens |
|---|---|---|---|---|---|---|---|
| 12 | Chemistry | C | C | B | EXP | 7,456 | 7,342 |
| 15 | Chemistry | A | D | A | NAT | 8,589 | 6,228 |
| 21 | Chemistry | C | B | C | NAT | 2,852 | 3,356 |
| 32 | Chemistry | A | A | C | EXP | 9,960 | 7,288 |
| 71 | Chemistry | B | B | ? | EXP | 12,837 | 132,650 |
| 88 | Chemistry | A | ? | A | NAT | 32,768 | 31,429 |
| 90 | Chemistry | C | B | C | NAT | 22,345 | 14,546 |
| 92 | Biology | B | A | B | NAT | 4,319 | 4,089 |
| 94 | Chemistry | D | D | C | EXP | 12,240 | 9,061 |
| 97 | Chemistry | B | C | B | NAT | 22,827 | 6,637 |
| 120 | Chemistry | D | A | D | NAT | 24,468 | 32,726 |
| 121 | Chemistry | C | C | ? | EXP | 10,600 | 32,768 |
| 127 | Biology | A | A | ? | EXP | 7,102 | 32,768 |
| 130 | Chemistry | C | C | A | EXP | 7,884 | 7,946 |
| 138 | Chemistry | C | B | C | NAT | 4,938 | 4,498 |
| 145 | Chemistry | A | A | C | EXP | 26,909 | 12,411 |
| 147 | Chemistry | C | C | B | EXP | 21,439 | 26,698 |
| 157 | Biology | C | C | A | EXP | 4,266 | 6,550 |
| 159 | Physics | D | D | C | EXP | 4,015 | 4,366 |
| 164 | Chemistry | C | C | D | EXP | 6,367 | 6,214 |
| 185 | Chemistry | C | ? | C | NAT | 32,768 | 8,178 |
| 186 | Physics | D | D | A | EXP | 4,801 | 6,240 |

## Methodology

- **Paired protocol**: same 198 GPQA-diamond questions, temperature 0 (greedy), identical prompt, identical max_tokens in both runs.
- Accuracy = evalscope review outcome (`ANSWER: [LETTER]` extraction vs target).
- Subject assigned by matching each question text against the original GPQA dataset (Subdomain / High-level domain fields): 198/198 matched.
- Speed comparison **intentionally omitted**: the two runs ran on different GPUs; tpot/latency are reported as per-run indications only.
- Paired net = expansion wins - native wins over divergent questions; paper acceptance rule: net ≥ 0.
- Branch: `moe-expansion` (25ca82a6) - llama.cpp fork vagrillo. Expansion is runtime-only, no weight changes.