#!/usr/bin/env python3
# Genera l'appendice del paper: RUN1209_comparison.md + RUN1209_appendix.html
# a partire da RUN1209_stats.json (prodotto da analyze_run1209.py)
import json
from pathlib import Path

BASE = Path(__file__).parent
S = json.load(open(BASE / "RUN1209_stats.json", encoding="utf-8"))
st = S["stats"]
paired = S["paired"]
subj = S["subj"]
n = S["n"]

LABELS = {
    "ds4_exp": "DS4-Flash-0731 exp (N=12, T=0.8, L28-42, decay→0.10)",
    "ds4_native": "DS4-Flash-0731 native (top-6)",
    "q35_exp": "Qwen3.6-35B exp (N=16, T=0.8, L25-39, decay→0.50)",
    "q35_native": "Qwen3.6-35B native (top-8)",
    "q27b": "Qwen3.8-27B (Q8, native)",
}
ORDER = ["ds4_exp", "ds4_native", "q35_exp", "q35_native", "q27b"]

acc_line = " | ".join(f"**{st[k]['acc']:.2f}%**" for k in ORDER)
hdr = " | ".join(LABELS[k].split(" (")[0] for k in ORDER)

md = []
md.append("# GPQA-Diamond benchmark — MoE expert expansion across models (paper appendix)\n")
md.append(f"> DeepSeek-V4-Flash-0731 (UD-IQ2_M) and Qwen3.6-35B-A3B (Q8_0), with and without MoE "
          f"expert expansion; Qwen3.8-27B (Q8) as a cross-model reference. "
          f"{n} questions, paired protocol, temperature 0, 32,768-token generation budget.\n")

md.append("## 1. Headline results\n")
md.append("| Configuration | accuracy | correct | tokens mean | tokens median | tokens p90 | truncated |")
md.append("|---|---|---|---|---|---|---|")
for k in ORDER:
    v = st[k]
    md.append(f"| {LABELS[k]} | **{v['acc']:.2f}%** | {v['n_ok']}/198 | {v['tok_mean']:,.0f} | "
              f"{v['tok_median']:,.0f} | {v['tok_p90']:,.0f} | {v['trunc']} |")
md.append("")

md.append("## 2. Expansion effect (paired, same questions)\n")
md.append("| Model | exp accuracy | native accuracy | Δ | exp wins | native wins | net |")
md.append("|---|---|---|---|---|---|---|")
for model, pk in [("DeepSeek-V4-Flash", "ds4"), ("Qwen3.6-35B", "q35")]:
    e, t = st[f"{pk}_exp"], st[f"{pk}_native"]
    p = paired[pk]
    we_, wn_ = p[0], p[1]
    md.append(f"| {model} | {e['acc']:.2f}% | {t['acc']:.2f}% | {e['acc']-t['acc']:+.2f} pts | "
              f"{we_} | {wn_} | **{we_-wn_:+d}** |")
md.append("\n- **DS4-Flash-0731**: net 0 — the expansion is neutral at these settings "
          "(the earlier net −4 was a budget artifact: native was capped at 16.3k).\n"
          "- **Qwen3.6-35B**: net +5 — the expansion wins (+2.52 pts), replicating the Q6_K result "
          "(net +4) on Q8_0.\n"
          "- Paper acceptance rule (net ≥ 0): satisfied for Qwen, not for DeepSeek at these settings.\n")

md.append("## 3. Does Qwen 35B + expansion reach the 27B?\n")
md.append("| Configuration | accuracy |")
md.append("|---|---|")
md.append(f"| Qwen3.8-27B (Q8) — newer gen, smaller | {st['q27b']['acc']:.2f}% |")
md.append(f"| Qwen3.6-35B native (top-8) | {st['q35_native']['acc']:.2f}% |")
md.append(f"| **Qwen3.6-35B expanded (N=16)** | **{st['q35_exp']['acc']:.2f}%** |")
md.append("\n**Yes**: the 35B with expansion (84.34%) surpasses the 27B Q8 reference (83.84%, +0.5 pts), "
          "while the native 35B (81.82%) falls 2.0 points short. The expansion closes the gap to the "
          "newer 27B model and overtakes it.\n")

md.append("## 4. Token consumption\n")
md.append("| Configuration | mean | median | p90 | total (198 q) | truncated |")
md.append("|---|---|---|---|---|---|")
for k in ORDER:
    v = st[k]
    md.append(f"| {LABELS[k]} | {v['tok_mean']:,.0f} | {v['tok_median']:,.0f} | {v['tok_p90']:,.0f} "
              f"| {v['tok_total']:,} | {v['trunc']} |")
md.append("\nExpansion at N=12/16 costs +4-5% mean tokens over native on Qwen; on DeepSeek-V4 the "
          "expansion is token-neutral (−5% mean, medians nearly equal). The 27B is the most verbose "
          "configuration (24 truncated at the 32k cap).\n")

md.append("## 5. Per-subject breakdown (accuracy %)\n")
md.append("| Subject | n | DS4 exp | DS4 nat | Q35 exp | Q35 nat | Q27B | net DS4 | net Q35 |")
md.append("|---|---|---|---|---|---|---|---|---|")
for dom, v in sorted(subj.items()):
    md.append(f"| {dom} | {v['n']} | {v['acc']['ds4_exp']:.1f} | {v['acc']['ds4_native']:.1f} "
              f"| {v['acc']['q35_exp']:.1f} | {v['acc']['q35_native']:.1f} | {v['acc']['q27b']:.1f} "
              f"| {v['net_ds4']:+d} | {v['net_q35']:+d} |")
md.append("\nExpansion nets are positive or neutral in every subject for both models.\n")

md.append("## 6. Methodology\n")
md.append("- Paired protocol: same 198 GPQA-diamond questions, temperature 0 (greedy), identical prompts, "
          "32,768-token generation budget in every run; evalscope review scoring.\n"
          "- Expansion is runtime-only (no weight changes): N = routed-expert budget above the native top-K, "
          "T = adaptive threshold on rank N/2, late-layer scoped, linear influence decay on the added ranks.\n"
          "- DS4 runs: unsloth UD-IQ2_M (~2-bit), server ctx 43,264. Qwen runs: unsloth Q8_0. "
          "Each model's runs on the same host; tpot/latency comparisons are valid within a model pair only.\n"
          "- Net = expansion wins − native wins over divergent questions; acceptance rule: net ≥ 0.\n")

md_text = "\n".join(md)
(BASE / "RUN1209_comparison.md").write_text(md_text, encoding="utf-8")
print(f"MD: {BASE / 'RUN1209_comparison.md'}")
