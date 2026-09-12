# RUN1209 — Final Comparative Report

## MoE Expert Expansion vs Native Routing on GPQA-Diamond: within-family (Qwen 35B MoE / 27B Dense) and cross-family (DeepSeek-V4-Flash-0731) evaluation

*Report date: 2026-09-12 · Dataset: GPQA-Diamond, 198/198 questions · Decoding: greedy (temperature 0) · Generation budget: 32,768 tokens/question · Serving: llama.cpp `moe-expansion` branch, `--parallel 4` · Scoring: evalscope review, `ANSWER: [LETTER]` extraction vs target.*

---

> ## ⚠️ Disclaimer — purpose and limits of RUN1209
>
> RUN1209 is designed to compare the benefits of **MoE expert expansion** in two directions:
>
> 1. **Within the same model family (Qwen):** Qwen3.6-35B-A3B (MoE, Q8_0) with expansion vs. its own native routing, and the **dense** Qwen3.8-27B (Q8) as an in-family reference point.
> 2. **Across model families:** the Qwen results against **DeepSeek-V4-Flash-0731** (UD-IQ2_M ≈ 2-bit), run with and without expansion.
>
> Within-family comparisons are strictly **paired**: same 198 questions, same prompt, same temperature, same token budget — the only changed variable is the routing configuration. Cross-family comparisons are **not** a controlled measurement of the expansion effect alone: they confound **quantization** (Qwen Q8_0 ≈ 8-bit vs. DeepSeek UD-IQ2_M ≈ 2-bit), **architecture/scale**, and **hardware** (all Qwen runs on a single RTX 5000 PRO 48 GB; all DeepSeek runs on a 4×GPU node with 128 GB of aggregate VRAM). Cross-family figures therefore describe *deployed configurations*, not intrinsic model quality.

**Data provenance.** Every figure in this report was recomputed directly from the raw per-question predictions and reviews in this run directory (`RunQ351209/`, `RunDS41209/`). Note that the summary file `RUN1209_results.md` contains one stale row (DeepSeek paired: "85.35% vs 80.30%, net +4"); the recount of this run's raw reviews gives **85.35% vs 85.35%, net 0** (6 expansion wins / 6 native wins / 163 both right / 23 both wrong), consistent with `RUN1209_stats.json` and with the "corrected equal-budget comparison" described in that file. This report uses the raw-review recount throughout.

---

## 1. Results at a glance

| Run | Model | Quant | Routing | Accuracy | Mean tokens | Mean s/question | tok/s |
|---|---|---|---|---:|---:|---:|---:|
| ds4_exp | DeepSeek-V4-Flash-0731 | UD-IQ2_M (~2-bit) | **expanded** (cap 12) | **85.35%** | 5,631 | 346.0 | 16.0 |
| ds4_native | DeepSeek-V4-Flash-0731 | UD-IQ2_M (~2-bit) | native top-6 | **85.35%** | 5,933 | 324.1 | 17.9 |
| q35_exp | Qwen3.6-35B-A3B | Q8_0 | **expanded** (cap 16) | 84.34% | 7,738 | 91.4 | 85.1 |
| q27b | Qwen3.8-27B (dense) | Q8_0 | native | 83.84% | 10,911 | 375.2 | 29.0 |
| q35_native | Qwen3.6-35B-A3B | Q8_0 | native top-8 | 81.82% | 7,416 | 75.8 | 98.7 |

Headline findings:

- **Expert expansion is clearly positive for Qwen3.6-35B** (+2.52 points, paired net **+5**, 12 wins vs 7 losses) at a +4.3% total token cost (**+9.5% on the matched both-correct questions** — see §3.4). It lifts the 35B MoE above the dense 27B reference (84.34% vs 83.84%) — something native routing does **not** achieve (81.82%).
- **Expansion is neutral for DeepSeek-V4-Flash** (paired net **0**, identical 85.35% accuracy), and actually used **5.1% fewer tokens** with 3 fewer truncations — a wash on quality, a small win on efficiency.
- **DeepSeek wins accuracy overall** (85.35%) despite 2-bit quantization, and is by far the most **token-efficient** (152 correct answers per Mtok vs 109 for Qwen3.6-35B).
- **Qwen wins throughput by a wide margin** (~85–99 tok/s vs ~16–18 tok/s), but note the two families ran on different hardware (single 48 GB GPU vs 4×GPU node).
- **Long-reasoning cliff:** on every configuration, accuracy collapses to 27–38% once reasoning exceeds 20k tokens; ~1/3 of that bucket never terminates (reasoning loops).

---

## 2. Configurations under test

| Run | Model | Quantization | Routing | Measured experts/token | Hardware |
|---|---|---|---|---|---|
| q35_exp | Qwen3.6-35B-A3B | Q8_0 | **expanded**: N=16, threshold 0.8, layers 25–39 of 40, decay 0.99→0.50 | ~11.6–13.9 (adaptive 4–16; native 8) | 1× RTX 5000 PRO 48 GB |
| q35_native | Qwen3.6-35B-A3B | Q8_0 | native top-8 | 8.0 | 1× RTX 5000 PRO 48 GB |
| q27b | Qwen3.8-27B (dense) | Q8_0 | native (reference) | — | 1× RTX 5000 PRO 48 GB |
| ds4_exp | DeepSeek-V4-Flash-0731 | UD-IQ2_M (≈2-bit) | **expanded**: N=12, threshold 0.8, layers 28–42 of 43, decay 0.99→0.10 | ~8.6–10.0 (adaptive 3–12; native 6) | 4× GPU, 128 GB VRAM |
| ds4_native | DeepSeek-V4-Flash-0731 | UD-IQ2_M (≈2-bit) | native top-6 | 6.0 | 4× GPU, 128 GB VRAM |

Common protocol: context ≈ 143k (Qwen) / 173k (DeepSeek) shared across 4 slots (~36k/43k per slot), batch 4, greedy decoding, `max_tokens = 32,768`, identical 198-question set and prompt template ("Think step by step… ANSWER: [LETTER]").

---

## 3. Qwen family analysis (RTX 5000 PRO 48 GB)

### 3.1 Accuracy

| Configuration | Correct / 198 | Accuracy |
|---|---:|---:|
| Qwen3.6-35B **expanded** (cap 16) | 167 | **84.34%** |
| Qwen3.8-27B dense (reference) | 166 | 83.84% |
| Qwen3.6-35B native (top-8) | 162 | 81.82% |

The expansion lifts the MoE model by **+2.52 points** over its own native routing and, crucially, makes the 35B MoE **overhaul the dense 27B** — the intended outcome of the experiment (a 35B-A3B MoE that matches/beats a 27B dense at a fraction of the compute per token).

### 3.2 Common questions (paired exp vs native)

| Pairing | exp wins | native wins | both right | both wrong | net |
|---|---:|---:|---:|---:|---:|
| q35 **exp** vs q35 native | **12** | 7 | 155 | 24 | **+5** |
| q35 exp vs q27b dense | 15 | 14 | 152 | 17 | +1 |

- The two Qwen3.6-35B runs agree on **179/198** questions (155 both right + 24 both wrong); the expansion changes the outcome on only **19 questions**, and wins 12 of them.
- **Where the wins come from:** 10 of the 12 expansion wins are **Chemistry** questions (plus 1 Physics, 1 Biology); native wins back 5 Chemistry, 1 Biology, 1 Physics. The expansion effect on Qwen is essentially a **Chemistry effect** (Chemistry: 70.97% → 76.34%).
- Across all five RUN1209 runs, **141 questions are solved by every configuration** and **12 by none**; only **45 questions** are contested at all — the benchmark's core is stable, and routing tweaks move the same marginal band of hard questions.

### 3.3 Subjects

| Subject | n | q35_exp | q35_native | q27b | Δ exp−native |
|---|---:|---:|---:|---:|---:|
| Chemistry | 93 | **76.34%** | 70.97% | 74.19% | **+5.37** |
| Physics | 86 | 95.35% | 95.35% | 95.35% | 0 |
| Biology | 19 | 73.68% | 73.68% | 78.95% | 0 |

Physics is saturated for the whole Qwen family (95.35% = 82/86, identical question set in all three runs). The dense 27B keeps a small edge on Biology (+5.27 over the 35B), while Chemistry — the largest and hardest subject — is exactly where wider routing pays off.

### 3.4 Token generation by reasoning-length class

Overall: q35_exp mean **7,738** (median 5,448, p90 16,751; total 1,532,034), q35_native mean **7,416** (median 5,403, p90 16,120; total 1,468,289), q27b mean **10,911** (median 6,758, p90 = 32,768 — cap-limited; total 2,160,347). Expansion costs **+4.3%** tokens on the paired 35B runs (+63,745 tokens total).

Accuracy by reasoning-length class (output tokens per question):

| Class | q35_exp n / acc | q35_native n / acc | q27b n / acc |
|---|---:|---:|---:|
| Brief (<2k) | 2 / 100% | 5 / 100% | 57 / 94.74% |
| Normal (2–5k) | 84 / 90.48% | 79 / 88.61% | 26 / 100% |
| Medium (5–10k) | 72 / 84.72% | 82 / 86.59% | 39 / 94.87% |
| Long (10–20k) | 27 / 85.19% | 20 / 60.00% | 37 / 97.30% |
| Very long (>20k) | 13 / **38.46%** | 12 / **33.33%** | 39 / **33.33%** |

Observations:

- The **dense 27B is much more verbose** (57 brief answers vs 2–5 for the MoE; 39 answers above 20k): it reasons long by default, and in the 2–20k range it is nearly flawless (95–100%). Its weakness is concentrated in the >20k bucket (33%), which contains its 24 truncations.
- The 35B MoE answers are shorter and its accuracy decays earlier along the length axis (Long bucket: 85.19% exp vs **60.0% native** — wider routing specifically rescues long-chain questions).
- For every Qwen configuration, the **>20k class is a failure zone** (33–38%): very long chains are dominated by non-termination (see 3.5).

**Matched-question check (both-correct ties).** Full-set token means mix different question outcomes: the expansion's 12 paired wins are, by construction, harder questions that require more reasoning. Restricting the comparison to the **155 questions both runs solved** removes that bias: expansion **6,472** vs native **5,913** mean tokens (**+9.5%**, +559 tokens/question) — higher than the +4.3% full-set figure — while the medians are nearly identical (**4,997 vs 5,065, −1.3%**): on half of the matched questions the expansion costs no extra tokens, and the mean gap is generated entirely in the long tail. A same-outcome restriction (179 questions, both right or both wrong) gives +7.5%, between the two.

### 3.5 Reasoning loops

- **Truncations at the 32,768 cap** (no final `ANSWER:` line emitted): q35_exp **6**, q35_native **5**, q27b **24**. **Every truncated answer was scored wrong** (0/N correct in all runs).
- These truncations are genuine **reasoning loops**, not slow-but-productive chains: the model cycles a hypothesis–contradiction pattern dozens of times. Example (q35_exp, question 131): the same three-line reasoning fragment repeats **~140 times** ("But 1,2,3,5 already gives 12, 6, 6. / This means the second compound must contribute 0 to the integrals? Impossible. / …") until the cap.
- A handful of answers terminate despite heavy repetition (compression ratio < 0.20): 0 on q35_exp, 1 on q35_native, 0 on q27b — loops that stop rarely succeed.
- **Loop-prone questions are shared**: question 88 trips *all five* RUN1209 runs; questions 81, 127, 147 and 185 trip the 27B and both DeepSeek runs. 32 distinct questions loop in at least one configuration — the loop trigger is mostly **question-intrinsic**, not routing-intrinsic.

### 3.6 Performance (same HW: RTX 5000 PRO 48 GB, batch 4)

| Metric | q35_exp | q35_native | q27b (dense) |
|---|---:|---:|---:|
| Mean latency / question | 91.4 s | 75.8 s | 375.2 s |
| Median latency / question | 63.7 s | 54.7 s | 227.7 s |
| p90 latency / question | 201.4 s | 159.1 s | 1,125.8 s |
| Mean tok/s (per question) | 85.1 | 98.7 | 29.0 |
| Server slot throughput | 85.7 tok/s | 99.1 tok/s | 29.6 tok/s |
| TTFT (mean) | 0.35 s | 0.36 s | 1.38 s |
| Total GPU time (Σ latencies) | 5.02 h | 4.17 h | 20.64 h |
| Time per correct answer | 108 s | 93 s | 448 s |

- **Wider routing costs ~13% throughput** (99.1 → 85.7 tok/s per slot) because the expanded layers route ~12–14 experts instead of 8. The accuracy gain is bought at +16 s mean latency and +0.85 GPU-hours.
- The **dense 27B is ≈3× slower per token** (29 vs 86–99 tok/s) and ~4–5× slower per question than the 35B MoE, despite having fewer total parameters — the expected dense-vs-sparse active-parameter effect (~27B active vs ~3B active) — and its verbosity multiplies the gap: **4.1× the total GPU time** of q35_exp for 0.5 fewer accuracy points.
- **Matched-question latency (155 both-correct ties):** 76.3 s (exp) vs 59.9 s (native), **+27%**, with per-question throughput unchanged (85.1 vs 99.0 tok/s) — on identical, correctly-solved questions the latency gap is generation volume (+9.5% tokens), not slower decoding.
- Best Qwen deployment in this run: **35B expanded for accuracy (84.34%), 35B native for speed (81.82% at 99 tok/s)**.

---

## 4. DeepSeek-V4-Flash-0731 analysis (4× GPU, 128 GB VRAM)

### 4.1 Accuracy

| Configuration | Correct / 198 | Accuracy |
|---|---:|---:|
| DS4 **expanded** (cap 12) | 169 | **85.35%** |
| DS4 native (top-6) | 169 | **85.35%** |

Identical accuracy — and the **best result of the entire RUN1209 exercise** — achieved at 2-bit quantization.

### 4.2 Common questions (paired exp vs native)

| Pairing | exp wins | native wins | both right | both wrong | net |
|---|---:|---:|---:|---:|---:|
| ds4 **exp** vs ds4 native | 6 | 6 | 163 | 23 | **0** |

- The two DeepSeek runs agree on **186/198** questions (163 both right + 23 both wrong) — the tightest pairing of the exercise. Expansion reshuffles 12 marginal questions and nets zero.
- Cross-family, on the same 198 questions: ds4_exp vs q35_exp → 16 vs 14 (net +2, both right 153); ds4_exp vs q35_native → 20 vs 13 (net +7).
- Both DeepSeek runs solve 163 questions identically; 23 resist both — again the contested band is thin.

### 4.3 Subjects

| Subject | n | ds4_exp | ds4_native | Δ exp−native |
|---|---:|---:|---:|---:|
| Chemistry | 93 | **81.72%** | 80.65% | +1.07 |
| Physics | 86 | 90.70% | 93.02% | −2.32 |
| Biology | 19 | 78.95% | 73.68% | +5.27 |

DeepSeek is the **best Chemistry model of the run by a wide margin** (81.7% vs 74–76% for Qwen) — remarkable at 2-bit — while Qwen keeps the Physics crown (95.35% vs 90.7–93.0%). The expansion's small per-subject swings (+5.3 Biology, −2.3 Physics) cancel out.

### 4.4 Token generation by reasoning-length class

Overall: ds4_exp mean **5,631** (median 1,580, p90 21,407), ds4_native mean **5,933** (median 1,443, p90 20,036). Expansion used **−5.1% tokens** (1,114,938 vs 1,174,801 total).

| Class | ds4_exp n / acc | ds4_native n / acc |
|---|---:|---:|
| Brief (<2k) | 109 / 94.50% | 112 / 93.75% |
| Normal (2–5k) | 27 / 92.59% | 28 / 89.29% |
| Medium (5–10k) | 27 / 88.89% | 20 / 85.00% |
| Long (10–20k) | 13 / 84.62% | 18 / 83.33% |
| Very long (>20k) | 22 / **27.27%** | 20 / **35.00%** |

Observations:

- DeepSeek's reasoning profile is the mirror image of Qwen's: **55% of questions answered in under 2k tokens** (vs 1–3% for Qwen3.6-35B), with graceful accuracy decay 94.5% → 84.6% across the first four classes.
- The >20k class is again the failure zone (27–35%): long-chain reasoning, not knowledge, is the universal bottleneck.
- Expansion moved answers from the 10–20k class into the 5–10k class (long: 18→13; medium: 20→27) while slightly *more* answers exceeded 20k (20→22) — a mixed signal — yet total tokens still fell 5.1%: routing became more decisive on easy questions.

### 4.5 Reasoning loops

- **Truncations at the 32,768 cap**: ds4_exp **9**, ds4_native **12** — all scored wrong. Expansion actually reduced loop-induced truncations by 3.
- Confirmed loops show the same signature as on Qwen: ds4_native question 79 repeats a single hypothesis line **156 times** ("Maybe output = 'sum of products of adjacent pairs'/2 + (sum of values)*? no."); question 127 cycles a DNA-sequence comparison ~96 times.
- Finished-but-repetition-heavy answers (compression ratio < 0.20) exist only on DeepSeek: 4 on ds4_exp (2 still correct), 2 on ds4_native (2 correct) — 2-bit decoding noise occasionally produces local repetition that the model still escapes.
- **Matched-question check (both-correct ties, n=163):** expansion **3,321** vs native **3,558** mean tokens (**−6.7%**) with median 1,169 vs 1,072 — on identical, correctly-solved questions the expanded routing still spends fewer tokens than native, confirming that the full-set −5.1% is not an artifact of question mix.
- Loop-prone questions overlap heavily with the other family (88 in all five runs; 81, 127, 147, 185 in both ds4 runs + q27b): loop triggers are **question-intrinsic**, consistent across families and quants.

### 4.6 Performance (same HW: 4× GPU, 128 GB VRAM, batch 4)

| Metric | ds4_exp | ds4_native |
|---|---:|---:|
| Mean latency / question | 346.0 s | 324.1 s |
| Median latency / question | 97.0 s | 78.5 s |
| p90 latency / question | 1,322.0 s | 1,131.5 s |
| Mean tok/s (per question) | 16.0 | 17.9 |
| Server slot throughput | 16.3 tok/s | 18.4 tok/s |
| TTFT (mean) | 1.09 s | 0.99 s |
| Total GPU time (Σ latencies) | 19.03 h | 17.82 h |
| Time per correct answer | 405 s | 380 s |

- Expansion costs **~11% generation throughput** (18.4 → 16.3 tok/s per slot, routing ~8.6–10 experts vs 6), but the 5.1% shorter answers claw most of it back: total GPU time rises only 1.2 h, and time-per-correct-answer is essentially unchanged (380 → 405 s).
- **Matched-question latency (163 both-correct ties):** 205.5 s (exp) vs 194.9 s (native), +5.4%, with tok/s 15.9 vs 17.8 — the ~11% routing throughput cost, on a −6.7% token volume, nets out to a near-zero latency difference on matched questions.
- The very high p90 (over 22 minutes) is driven by loop questions burning the full 32k budget — 9–12 questions per run dominate the tail latency.

---

## 5. Cross-family comparison: Qwen vs DeepSeek-V4-Flash

**Quantization caveat (central to this comparison):** Qwen3.6-35B runs at **Q8_0 (~8-bit, near-lossless)** while DeepSeek-V4-Flash runs at **UD-IQ2_M (~2-bit, aggressive)**. DeepSeek matching or beating Qwen's accuracy *from a 2-bit model* is the single most important cross-family datum: it suggests the full-precision DeepSeek headroom is substantially higher, while Qwen's Q8 numbers are already close to its ceiling.

| Dimension | Qwen3.6-35B-A3B (best: exp) | DeepSeek-V4-Flash-0731 (exp = native) | Edge |
|---|---|---|---|
| Accuracy (198) | 84.34% (native 81.82%) | **85.35%** | DeepSeek (+1.0) |
| Quantization | Q8_0 (~8-bit) | UD-IQ2_M (~2-bit) | DeepSeek's result is achieved under a 4× tighter budget |
| Expansion effect | **+2.52 pts, net +5** | ±0.0, net 0 | Expansion pays only on Qwen |
| Total tokens | 1.532 M (exp) / 1.468 M (native) | **1.115 M** (exp) / 1.175 M (native) | DeepSeek −27% tokens vs q35_exp |
| Reasoning style | long chains (median 5.4k) | brief chains (median 1.4–1.6k) | different families, same tail behaviour |
| Correct / Mtok | 109.0 | **151.6** | DeepSeek +39% token efficiency |
| Truncations (loops) | 6 (exp) / 5 (native) / 24 (27B) | 9 (exp) / 12 (native) | Qwen3.6-35B cleaner |
| Chemistry | 76.3% | **81.7%** | DeepSeek |
| Physics | **95.3%** | 90.7–93.0% | Qwen |
| Throughput (own HW) | **85–99 tok/s** (1× 48 GB) | 16–18 tok/s (4× GPU, 128 GB) | Qwen ≈ 5× faster |
| Latency / question (own HW) | 75–91 s | 324–346 s | Qwen ≈ 3.8× faster |
| Time / correct answer (own HW) | **108 s** | 405 s | Qwen, on this hardware |
| HW requirement | 48 GB single GPU (Q8, 35B) | 128 GB multi-GPU (2-bit) | Qwen: 37% of the VRAM at 4× the bit budget |

Reading of the table:

1. **Accuracy vs quantization.** DeepSeek leads by 1.0 points at ~2-bit; at matched precision the gap would likely widen in its favour. Qwen's counter-argument is deployment economics: it delivers 84% at 8-bit on a single 48 GB card.
2. **Expansion is family-dependent.** The paper's acceptance rule (net ≥ 0 vs native) is clearly met by Qwen3.6-35B (**+5**) and met only marginally by DeepSeek (**0**: 6 wins / 6 losses, no accuracy change). At 2-bit the router logits beyond the native top-6 appear to carry more noise than signal — adding experts neither helps nor hurts, it just (slightly) shortens answers. This is consistent with the model-dependence warning in the ds4 documentation.
3. **Different verbosity strategies, same failure mode.** DeepSeek answers briefly and stays accurate up to 10k tokens; Qwen reasons long and stays accurate up to 20k (with expansion rescuing the 10–20k band). Both collapse identically beyond 20k, mostly via the same loop-prone questions (88, 81, 127, 147, 185).
4. **Throughput is hardware-dominated.** The 5× tok/s advantage of Qwen reflects both sparsity (3B active params vs DeepSeek's larger active set) and the single-GPU-vs-4-GPU setup; it should not be read as an architecture verdict across hosts.

---

## 6. Conclusions

1. **The expansion's benefit is real but model-dependent — the central result of RUN1209.** On Qwen3.6-35B-A3B it delivers **+2.52 accuracy points (paired net +5)** for +4.3% tokens overall (+9.5% on the matched both-correct questions, with flat medians) and −13% throughput, and it is the *only* configuration that pushes the 35B MoE past the dense Qwen3.8-27B (84.34% vs 83.84%). The paper acceptance rule (net ≥ 0) is therefore **satisfied within the Qwen family**. On DeepSeek-V4-Flash the rule is met only at the boundary (**net 0**, 85.35% both ways): at 2-bit, expansion neither helps nor regresses — usable, but not a lever.
2. **Where Qwen gains is Chemistry** (70.97 → 76.34; 10 of 12 expansion wins), while Physics is saturated at 95.35% regardless of routing. DeepSeek, conversely, is the strongest Chemistry model overall (81.7%) and concedes Physics to Qwen.
3. **The dense 27B reference validates the MoE-with-expansion thesis.** The 27B is accurate (83.84%) but ≈3× slower per token, 4.1× more GPU-hungry, 47% more verbose, and the most loop-prone configuration (24 truncations). A 35B-A3B with expanded routing beats it on accuracy at ~24% of the GPU cost.
4. **Very long reasoning is the universal failure mode.** Accuracy above 20k tokens is 27–38% everywhere; every truncated answer is wrong; five questions (88, 81, 127, 147, 185) loop across families, quants and routing modes. A repetition-detection early-stop (or a loop-aware sampler) would reclaim several points of accuracy and a large share of tail latency at zero quality cost — the most actionable engineering outcome of this run.
5. **Family verdict.** For accuracy per token and Chemistry strength, DeepSeek-V4-Flash-0731 leads — an impressive result for a 2-bit deployment, and its headroom at higher precision is untested here. For deployable throughput and accuracy-per-second on modest hardware, Qwen3.6-35B-A3B **with expansion** is the best configuration in RUN1209: 84.34% at ~86 tok/s on a single 48 GB GPU. The two families are complementary rather than competing on this benchmark; the expansion technique itself earns its place in the Qwen stack and remains optional (harmless) for DeepSeek.

---

### Appendix: file map

| Artifact | Path |
|---|---|
| Qwen predictions/reviews | `RunQ351209/root/gpqa-{exp,native,27BQ8}/…` |
| DeepSeek predictions/reviews | `RunDS41209/root/gpqa-ds4-{exp,native}/…` |
| Server logs (moe counters, slot timings) | `Run*/root/logs/server-*.log` |
| Aggregated stats | `RUN1209_stats.json`, `final_analysis.json` |
| Recompute script | `final_analysis.py` |
| HTML version of this report (paper appendix) | `finalcompariso1209.html` |
