# DeepSeek-V4-Flash-0731 — GPQA-Diamond: MoE expansion vs native routing

> Model: DeepSeek-V4-Flash-0731 (unsloth UD-IQ2_M, ~91GB) · 198 questions, paired, temperature 0 · **Expansion: N=12, T=0.8, layers 30-42, decay 0.99→0.50** · Native: top-6 · llama.cpp branch `moe-expansion`

> ⚠️ **Budget asymmetry in this pair of runs**: expansion ran with `max_tokens=32,768`, native with `max_tokens=16,384` (server ctx 16,384). A native-32k run is in progress; until then, the **clean subset (section 3) is the only budget-fair comparison**.

## 1. Headline (all 198, as scored)

| Metric | Expansion (N=12) | Native (top-6) | Δ |
|---|---|---|---|
| accuracy | **85.35%** (169/198) | 80.30% (159/198) | +5.05 pts |
| paired net | **+10** (14 wins / 4 losses) | - | |
| tokens - mean | 5,631 | 4,290 | +31.3% |
| tokens - median | 1,580 | 1,562 | - |
| truncated (stop=length) | 9 | 25 | - |
| total tokens | 1,114,938 | 849,440 | +31.3% |

Measured routed experts/token on the expansion run: **~9.7** (native 6, adaptive range 3-12)

## 2. The budget asymmetry (read this before the net)

The two runs had **different generation budgets** (32,768 vs 16,384). 30 questions were answered by the expansion beyond the native budget (response > 16.4k tokens) or truncated by the native run. On that hard tail:

| Subset | n | exp accuracy | native accuracy |
|---|---|---|---|
| clean (both complete ≤ 16.4k) | 168 | 92.3% | 93.5% |
| excluded hard tail (exp > 16.4k or nat truncated) | 30 | 46.7% | 6.7% |

**Interpretation:** the raw-198 view (+10) is dominated by the hard tail, where the 32k budget lets the expansion finish answers the native run cuts at 16.4k (46.7% vs 6.7% there). On the budget-fair clean subset (168 questions) the two configurations are close (92.3% vs 93.5%, net -2).

## 3. Budget-fair paired comparison (clean subset)

n = 168 questions answered completely by **both** configurations within the native 16,384-token budget:

- both correct: 153
- both wrong: 9
- expansion wins: 2
- native wins: 4
- **net (exp − nat): -2** — within noise at this sample size

| Metric | Expansion | Native |
|---|---|---|
| accuracy | 92.26% | 93.45% |
| tokens - mean | 2,477 | 2,380 |
| tokens - median | 1,072 | 1,078 |
| tokens - p90 | 7,209 | 6,684 |

Token consumption on completable questions is essentially identical — the expansion does not lengthen answers here.

## 4. Per-subject (clean subset)

| Subject | n | exp acc | native acc | net | exp mean tok | native mean tok | exp med tok | native med tok |
|---|---|---|---|---|---|---|---|---|
| Biology | 17 | 82.4% (14) | 88.2% (15) | 🔴 -1 | 2,582 | 1,368 | 710 | 534 |
| Chemistry | 67 | 95.5% (64) | 92.5% (62) | 🟢 +2 | 3,761 | 3,706 | 2,885 | 2,351 |
| Physics | 84 | 91.7% (77) | 95.2% (80) | 🔴 -3 | 1,432 | 1,526 | 661 | 594 |

## 5. Subcategory detail (clean subset)

| Subcategory | n | exp acc | native acc | net |
|---|---|---|---|---|
| Astrophysics | 13 | 76.9% (10) | 100.0% (13) | 🔴 -3 |
| Chemistry (general) | 17 | 100.0% (17) | 100.0% (17) | ⚪ +0 |
| Condensed Matter Physics | 1 | 100.0% (1) | 100.0% (1) | ⚪ +0 |
| Electromagnetism and Photonics | 6 | 100.0% (6) | 100.0% (6) | ⚪ +0 |
| Genetics | 4 | 75.0% (3) | 75.0% (3) | ⚪ +0 |
| High-energy particle physics | 14 | 92.9% (13) | 92.9% (13) | ⚪ +0 |
| Inorganic Chemistry | 1 | 100.0% (1) | 100.0% (1) | ⚪ +0 |
| Molecular Biology | 13 | 84.6% (11) | 92.3% (12) | 🔴 -1 |
| Optics and Acoustics | 1 | 100.0% (1) | 100.0% (1) | ⚪ +0 |
| Organic Chemistry | 49 | 93.9% (46) | 89.8% (44) | 🟢 +2 |
| Physics (general) | 19 | 89.5% (17) | 89.5% (17) | ⚪ +0 |
| Quantum Mechanics | 23 | 100.0% (23) | 100.0% (23) | ⚪ +0 |
| Relativistic Mechanics | 7 | 85.7% (6) | 85.7% (6) | ⚪ +0 |

## 6. Divergent questions on the clean subset

| # | Subject | Target | Exp answer | Native answer | Winner | exp tok | nat tok |
|---|---|---|---|---|---|---|---|
| answer the following multiple choice question. the | Biology | C | B | C | 🔴 NAT | 11,597 | 5,599 |
| answer the following multiple choice question. the | Physics | D | A | D | 🔴 NAT | 2,957 | 3,438 |
| answer the following multiple choice question. the | Chemistry | C | C | D | 🟢 EXP | 13,587 | 6,265 |
| answer the following multiple choice question. the | Physics | C | D | C | 🔴 NAT | 368 | 204 |
| answer the following multiple choice question. the | Physics | B | D | B | 🔴 NAT | 3,751 | 1,556 |
| answer the following multiple choice question. the | Chemistry | C | C | B | 🟢 EXP | 3,511 | 6,349 |

## 7. Token budget considerations

- **The 16.4k cap costs the native run real accuracy**: 30 questions (of 198) produced > 16.4k-token responses under expansion; native scored 6.7% on them (mostly truncations at 16.4k) vs expansion 46.7% with its 32k budget. Part of that gap is genuine budget value, part may be routing quality.
- **A native-32k run is in progress** and will be the definitive paired comparison at equal budgets.
- **Budget rule of thumb from this data**: DeepSeek-V4-Flash reasoning on GPQA needs > 16k tokens on roughly 15% of questions; a 16.4k cap truncates those.
- On completable questions the expansion consumes essentially the same tokens as native (see section 3) — the cost of N=12 is per-token speed, not answer length.

## 8. Cross-model: DeepSeek-V4-Flash vs Qwen3.6-35B (both expanded)

<div class="box note">Same 198 questions, both with the expansion active (DS4: N=12/T=0.8/L30-42; Qwen: N=20/T=0.8/L29-39). Cross-model accuracy is a model comparison, not an expansion-effect measurement. Quants differ (IQ2_M vs Q6_K); hardware differs (t/s not comparable).</div>

| Config | accuracy on the 198 common questions |
|---|---|
| DS4-Flash-0731 (IQ2_M) expanded | 85.35% |
| Qwen3.6-35B-A3B (Q6_K) expanded | 85.35% |

Agreement: both correct 153 · DS4-only 16 · Qwen-only 16 · both wrong 13.

Native references (separate runs, same benchmark): DS4-Flash native 80.30% · Qwen3.6 native 83.33%.

## 9. Methodology

- Paired protocol: same 198 GPQA-diamond questions, temperature 0 (greedy), identical prompt.
- Expansion: N=12, T=0.8, layers 30-42 of 43, decay 0.99→0.50; native: top-6. Same host, same quant (UD-IQ2_M), same server build.
- Accuracy = evalscope review outcome (ANSWER: [LETTER] extraction vs target).
- Budgets differ by design of this pair of runs (exp 32k vs nat 16.4k): section 2. A native-32k run is in progress for the definitive equal-budget comparison.
- Clean subset excludes questions where either configuration hit its budget, to compare routing quality at equal budget.
- Cross-reference: Qwen3.6-35B-A3B (Q6_K) same question set: expansion +2.02 pts, net +4 ([report](../GPQA/report_gpqa_moe.md)).