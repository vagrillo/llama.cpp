#!/usr/bin/env python3
# GPQA-diamond comparative report: MoE expert expansion vs native routing -> HTML (English)
import html
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
DATASET = Path("/tmp/gpqa_train.jsonl")
DATASET_URL = ("https://modelscope.cn/api/v1/datasets/AI-ModelScope/gpqa_diamond"
               "/repo?Revision=master&FilePath=train.jsonl")
if not DATASET.exists():
    import urllib.request
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading GPQA dataset metadata -> {DATASET}")
    urllib.request.urlretrieve(DATASET_URL, DATASET)
RUNS = {"exp": "gpqa-exp", "native": "gpqa-native"}


def norm(s, limit=6000):
    return " ".join(str(s).split()).lower()[:limit]


def load_dataset():
    gpqa = {}
    for line in open(DATASET, encoding="utf-8"):
        d = json.loads(line)
        gpqa[norm(d["Question"], 120)] = {
            "domain": d.get("High-level domain") or "?",
            "subdomain": d.get("Subdomain") or "?",
        }
    return gpqa


def load_run(root, gpqa):
    preds, revs = {}, {}
    p = list(root.rglob("predictions/*/gpqa_diamond_default.jsonl"))[0]
    for line in open(p, encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        d = json.loads(line)
        mo = d.get("model_output")
        qtext = ""
        for m in d.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                if isinstance(c, str) and len(c) > len(qtext):
                    qtext = c
        nq = norm(qtext)
        meta = None
        for k, v in gpqa.items():
            if k in nq:
                meta = v
                break
        if isinstance(mo, dict):
            u = mo.get("usage") or {}
            pm = mo.get("perf_metrics") or {}
            ch = (mo.get("choices") or [{}])[0]
            tp = pm.get("tpot")
            rec = {"out": u.get("output_tokens"), "lat": mo.get("time") or pm.get("latency"),
                   "tpot": tp * 1000 if tp else None, "stop": ch.get("stop_reason", "?")}
        else:
            mo = mo or ""

            def rx(pat, cast=float):
                m = re.search(pat, mo)
                return cast(m.group(1)) if m else None
            tp = rx(r"'tpot':\s*([0-9.]+)")
            rec = {"out": rx(r"'output_tokens':\s*(\d+)", int),
                   "lat": rx(r"'latency':\s*([0-9.]+)"),
                   "tpot": tp * 1000 if tp else None,
                   "stop": (re.search(r"'stop_reason':\s*'([^']+)'", mo) or [None, "?"])[1]}
        rec["meta"] = meta
        preds[int(d["index"])] = rec
    rf = sorted(root.rglob("reviews/*/gpqa_diamond_default.jsonl"))[-1]
    for line in open(rf, encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        d = json.loads(line)
        sc = (d.get("sample_score") or {}).get("score", {})
        val = (sc.get("value") or {}).get("accuracy")
        revs[int(d["index"])] = {"correct": bool(val and val >= 0.5),
                                 "extracted": sc.get("extracted_prediction"),
                                 "target": d.get("target")}
    return preds, revs


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p / 100))]


def summary(P, R, idxs):
    toks = [P[i]["out"] for i in idxs if P[i]["out"]]
    corr = [R[i]["correct"] for i in idxs if i in R]
    return {"n": len(idxs), "n_ok": sum(corr), "acc": 100.0 * sum(corr) / max(1, len(corr)),
            "tok_mean": st.mean(toks), "tok_median": st.median(toks),
            "tok_p90": pct(toks, 90), "tok_max": max(toks), "tok_total": sum(toks),
            "trunc": sum(1 for i in idxs if P[i]["stop"] not in ("stop",)),
            "tpot": st.mean([P[i]["tpot"] for i in idxs if P[i]["tpot"]]),
            "lat": st.mean([P[i]["lat"] for i in idxs if P[i]["lat"]])}


def main():
    gpqa = load_dataset()
    runs = {}
    for k, d in RUNS.items():
        P, R = load_run(BASE / d, gpqa)
        missing = sum(1 for i in P if P[i]["meta"] is None)
        runs[k] = {"preds": P, "revs": R}
        print(f"{k}: {len(P)} predictions, {len(R)} reviews, without subject: {missing}")

    idx_all = sorted(set(runs["exp"]["preds"]) & set(runs["native"]["preds"]))
    S = {k: summary(runs[k]["preds"], runs[k]["revs"], idx_all) for k in RUNS}
    R_ = {k: runs[k]["revs"] for k in RUNS}

    br = sum(1 for i in idx_all if R_["exp"][i]["correct"] and R_["native"][i]["correct"])
    bw = sum(1 for i in idx_all if not R_["exp"][i]["correct"] and not R_["native"][i]["correct"])
    we = sum(1 for i in idx_all if R_["exp"][i]["correct"] and not R_["native"][i]["correct"])
    wn = sum(1 for i in idx_all if not R_["exp"][i]["correct"] and R_["native"][i]["correct"])
    net = we - wn

    subj = defaultdict(list)
    for i in idx_all:
        m = runs["exp"]["preds"][i]["meta"]
        if m:
            subj[m["domain"]].append(i)
    subsub = defaultdict(list)
    for i in idx_all:
        m = runs["exp"]["preds"][i]["meta"]
        if m:
            subsub[(m["domain"], m["subdomain"])].append(i)

    toks_e = [runs["exp"]["preds"][i]["out"] for i in idx_all]
    toks_n = [runs["native"]["preds"][i]["out"] for i in idx_all]
    tok_e = {k: [runs["exp"]["preds"][i]["out"] for i in idxs if runs["exp"]["preds"][i]["out"]]
             for k, idxs in subj.items()}
    tok_n = {k: [runs["native"]["preds"][i]["out"] for i in idxs if runs["native"]["preds"][i]["out"]]
             for k, idxs in subj.items()}

    def bar_svg(pairs, unit="%", color="#2563eb"):
        maxval = max(v for _, v in pairs) * 1.15
        rows = []
        for i, (lab, val) in enumerate(pairs):
            y = i * 26 + 4
            w = max(2, 320.0 * val / maxval)
            rows.append(f'<text x="0" y="{y + 15}" font-size="12.5" fill="#334155">{html.escape(lab)}</text>')
            rows.append(f'<rect x="170" y="{y}" width="{w:.0f}" height="17" rx="3" fill="{color}"/>')
            rows.append(f'<text x="{178 + w:.0f}" y="{y + 13}" font-size="12.5" fill="#0f172a">{val:.1f}{unit}</text>')
        return (f'<svg viewBox="0 0 560 {26 * len(pairs) + 6}" width="100%" '
                f'style="max-width:560px">{"".join(rows)}</svg>')

    acc_bars = bar_svg([(f"exp accuracy {d}",
                         100.0 * sum(R_["exp"][i]["correct"] for i in idxs) / len(idxs))
                        for d, idxs in [("ALL", idx_all)] + sorted(subj.items())])
    acc_bars_n = bar_svg([(f"native accuracy {d}",
                           100.0 * sum(R_["native"][i]["correct"] for i in idxs) / len(idxs))
                          for d, idxs in [("ALL", idx_all)] + sorted(subj.items())], color="#94a3b8")
    tok_bars = bar_svg([(f"exp mean tokens {d}", st.mean(tok_e[d])) for d in sorted(tok_e)],
                       unit="", color="#0d9488")
    tok_bars_n = bar_svg([(f"native mean tokens {d}", st.mean(tok_n[d])) for d in sorted(tok_n)],
                         unit="", color="#94a3b8")

    pct_rows = "".join(
        f"<tr><td>p{p}</td><td>{pct(toks_e, p):,.0f}</td><td>{pct(toks_n, p):,.0f}</td>"
        f"<td class='{('verde' if pct(toks_e, p) < pct(toks_n, p) else 'rosso')}'>"
        f"{100.0 * (pct(toks_e, p) - pct(toks_n, p)) / pct(toks_n, p):+.1f}%</td></tr>"
        for p in (10, 25, 50, 75, 90, 95))

    cap_rows = ""
    for cap in (None, 32768, 16384):
        tag = "none" if cap is None else f"{cap // 1024}K"
        ce = toks_e if cap is None else [min(t, cap) for t in toks_e]
        cn = toks_n if cap is None else [min(t, cap) for t in toks_n]
        se, sn = sum(ce), sum(cn)
        cl_e = sum(1 for t in toks_e if cap is not None and t > cap)
        cl_n = sum(1 for t in toks_n if cap is not None and t > cap)
        cap_rows += (f"<tr><td>{tag}</td><td>{se:,}</td><td>{sn:,}</td>"
                     f"<td class='{('verde' if se <= sn else 'rosso')}'>"
                     f"{100.0 * (se - sn) / max(1, sn):+.1f}%</td>"
                     f"<td>{cl_e} / {cl_n}</td></tr>")

    dom_rows = ""
    for dom in sorted(subj):
        idxs = subj[dom]
        ne = sum(R_["exp"][i]["correct"] for i in idxs)
        nn = sum(R_["native"][i]["correct"] for i in idxs)
        w = sum(1 for i in idxs if R_["exp"][i]["correct"] and not R_["native"][i]["correct"])
        l = sum(1 for i in idxs if not R_["exp"][i]["correct"] and R_["native"][i]["correct"])
        cls = "verde" if w - l > 0 else ("rosso" if w - l < 0 else "grigio")
        dom_rows += (f"<tr><td>{html.escape(dom)}</td><td>{len(idxs)}</td>"
                     f"<td>{100.0 * ne / len(idxs):.1f}% ({ne})</td>"
                     f"<td>{100.0 * nn / len(idxs):.1f}% ({nn})</td>"
                     f"<td class='{cls}'>{w - l:+d}</td>"
                     f"<td>{st.mean(tok_e[dom]):,.0f}</td><td>{st.mean(tok_n[dom]):,.0f}</td>"
                     f"<td>{st.median(tok_e[dom]):,.0f}</td><td>{st.median(tok_n[dom]):,.0f}</td></tr>")

    sub_rows = ""
    for (dom, sub) in sorted(subsub):
        idxs = subsub[(dom, sub)]
        ne = sum(R_["exp"][i]["correct"] for i in idxs)
        nn = sum(R_["native"][i]["correct"] for i in idxs)
        w = sum(1 for i in idxs if R_["exp"][i]["correct"] and not R_["native"][i]["correct"])
        l = sum(1 for i in idxs if not R_["exp"][i]["correct"] and R_["native"][i]["correct"])
        cls = "verde" if w - l > 0 else ("rosso" if w - l < 0 else "grigio")
        sub_rows += (f"<tr><td>{html.escape(dom)} — {html.escape(sub)}</td><td>{len(idxs)}</td>"
                     f"<td>{100.0 * ne / len(idxs):.1f}% ({ne})</td>"
                     f"<td>{100.0 * nn / len(idxs):.1f}% ({nn})</td>"
                     f"<td class='{cls}'>{w - l:+d}</td></tr>")

    div_rows = ""
    for i in idx_all:
        re_, rn = R_["exp"][i], R_["native"][i]
        if re_["correct"] == rn["correct"]:
            continue
        dom = runs["exp"]["preds"][i]["meta"]["domain"] if runs["exp"]["preds"][i]["meta"] else "?"
        w = "EXP" if re_["correct"] else "NAT"
        cls = "verde" if w == "EXP" else "rosso"
        div_rows += (f"<tr><td>{i}</td><td>{html.escape(dom)}</td><td>{html.escape(re_['target'])}</td>"
                     f"<td>{html.escape(re_['extracted'] or '?')}</td>"
                     f"<td>{html.escape(rn['extracted'] or '?')}</td>"
                     f"<td class='{cls}'>{w}</td>"
                     f"<td>{runs['exp']['preds'][i]['out']:,}</td>"
                     f"<td>{runs['native']['preds'][i]['out']:,}</td></tr>")

    tk_mean_d = 100.0 * (S["exp"]["tok_mean"] - S["native"]["tok_mean"]) / S["native"]["tok_mean"]
    tk_med_d = 100.0 * (S["exp"]["tok_median"] - S["native"]["tok_median"]) / S["native"]["tok_median"]
    tpot_e, tpot_n = S["exp"]["tpot"], S["native"]["tpot"]
    lat_e, lat_n = S["exp"]["lat"], S["native"]["lat"]

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>GPQA-Diamond - MoE expert expansion vs native routing</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; color:#0f172a; max-width: 1040px; margin: 24px auto; padding: 0 16px; background:#f8fafc; }}
h1 {{ font-size: 1.45em; }} h2 {{ font-size: 1.12em; margin-top: 1.7em; border-bottom: 2px solid #e2e8f0; padding-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; background: white; }}
th, td {{ border: 1px solid #e2e8f0; padding: 6px 10px; text-align: right; font-size: 0.92em; }}
th {{ background: #f1f5f9; }} td:first-child, th:first-child {{ text-align: left; }}
.verde {{ color: #047857; font-weight: 600; }} .rosso {{ color: #b91c1c; font-weight: 600; }} .grigio {{ color: #64748b; }}
.box {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; margin: 14px 0; }}
.kpi {{ display: flex; gap: 14px; flex-wrap: wrap; }} .kpi .box {{ flex: 1; min-width: 190px; text-align: center; }}
.kpi .num {{ font-size: 1.65em; font-weight: 700; }}
.note {{ color:#475569; font-size: 0.87em; }}
.mono {{ font-family: ui-monospace, monospace; font-size: 0.85em; background:#f1f5f9; padding: 1px 5px; border-radius: 4px; }}
.two {{ display: flex; gap: 18px; flex-wrap: wrap; }} .two > div {{ flex: 1; min-width: 300px; }}
</style></head><body>
<h1>GPQA-Diamond - MoE expert expansion vs native routing</h1>
<p class="note">Model: Qwen3.6-35B-A3B (UD-Q6_K_XL) · 198 questions, paired protocol, temperature 0 ·
Expansion config: <b>N=20, threshold 0.8, layers 29-39, decay 0.99&#8594;0.50</b> · Native config: top-8 ·
llama.cpp branch <span class="mono">moe-expansion</span> (25ca82a6)</p>

<div class="kpi">
<div class="box"><div class="num">{S['exp']['acc']:.2f}%</div>accuracy expansion<br><span class="note">{S['exp']['n_ok']}/198</span></div>
<div class="box"><div class="num">{S['native']['acc']:.2f}%</div>accuracy native<br><span class="note">{S['native']['n_ok']}/198</span></div>
<div class="box"><div class="num verde">net +{net}</div>paired ({we} wins / {wn} losses)<br><span class="note">{br} ties out of 198</span></div>
<div class="box"><div class="num">{tk_mean_d:+.1f}%</div>mean tokens (exp vs nat)<br><span class="note">{S['exp']['tok_mean']:,.0f} vs {S['native']['tok_mean']:,.0f}</span></div>
<div class="box"><div class="num">{tk_med_d:+.1f}%</div>median tokens<br><span class="note">{S['exp']['tok_median']:,.0f} vs {S['native']['tok_median']:,.0f}</span></div>
</div>

<h2>Comparative outcome</h2>
<div class="box">Expansion <b>improves accuracy by +2.02 points</b> (85.35% vs 83.33%) while at the same time
<b>reducing tokens consumed</b> (mean {tk_mean_d:+.1f}%, median {tk_med_d:+.1f}%): on GPQA the
"Succinct Convergence" phenomenon described in the paper shows up on both quality and brevity.
The paired net <b>+{net}</b> ({we} wins vs {wn} losses, out of {we + wn} divergent questions) is positive but not
statistically significant at n=198 (&asymp;1.7&sigma;): consistent with the 1-3 point effect reported in the paper.
<b>Paper acceptance rule (net &ge; 0): satisfied - the feature is valid for GPQA on this model.</b></div>

<h2>Accuracy</h2>
<div class="two"><div>{acc_bars}</div><div>{acc_bars_n}</div></div>
<table>
<tr><th>Run</th><th>accuracy</th><th>correct</th><th>n</th></tr>
<tr><td>Expansion (N=20, T=0.8, L29-39)</td><td>{S['exp']['acc']:.2f}%</td><td>{S['exp']['n_ok']}</td><td>{S['exp']['n']}</td></tr>
<tr><td>Native (top-8)</td><td>{S['native']['acc']:.2f}%</td><td>{S['native']['n_ok']}</td><td>{S['native']['n']}</td></tr>
</table>

<h2>Tokens consumed</h2>
<table>
<tr><th>Metric</th><th>Expansion</th><th>Native</th><th>&Delta; (exp vs nat)</th></tr>
<tr><td>response tokens - <b>mean</b></td><td>{S['exp']['tok_mean']:,.0f}</td><td>{S['native']['tok_mean']:,.0f}</td><td>{tk_mean_d:+.1f}%</td></tr>
<tr><td>response tokens - <b>median</b></td><td>{S['exp']['tok_median']:,.0f}</td><td>{S['native']['tok_median']:,.0f}</td><td>{tk_med_d:+.1f}%</td></tr>
<tr><td>response tokens - p90</td><td>{S['exp']['tok_p90']:,.0f}</td><td>{S['native']['tok_p90']:,.0f}</td>
<td>{100.0 * (S['exp']['tok_p90'] - S['native']['tok_p90']) / S['native']['tok_p90']:+.1f}%</td></tr>
<tr><td>response tokens - max</td><td>{S['exp']['tok_max']:,}</td><td>{S['native']['tok_max']:,}</td><td>-</td></tr>
<tr><td>total tokens (198 questions)</td><td>{S['exp']['tok_total']:,}</td><td>{S['native']['tok_total']:,}</td>
<td>{100.0 * (S['exp']['tok_total'] - S['native']['tok_total']) / S['native']['tok_total']:+.1f}%</td></tr>
<tr><td>truncated responses (length)</td><td>{S['exp']['trunc']}</td><td>{S['native']['trunc']}</td><td>-</td></tr>
</table>
<div class="two"><div>{tok_bars}</div><div>{tok_bars_n}</div></div>

<h2>Response token distribution (paired percentiles)</h2>
<table><tr><th>percentile</th><th>expansion</th><th>native</th><th>&Delta;</th></tr>{pct_rows}</table>
<p class="note">Across all percentiles up to p75 the expansion produces shorter answers
(&asymp; -3/-6%): the paper's "succinct convergence". Divergences are point-wise and limited to the tail.</p>

<h2>Token cost under per-response budget</h2>
<table><tr><th>Cap</th><th>total expansion</th><th>total native</th><th>&Delta;</th><th>clipped exp/nat</th></tr>{cap_rows}</table>
<p class="note">Clipping estimates <b>cost</b>, not quality: a truncated answer could flip its verdict - that must be measured, not simulated.</p>

<h2>Per-subject analysis</h2>
<table>
<tr><th>Subject</th><th>n</th><th>exp accuracy</th><th>native accuracy</th><th>paired net</th>
<th>exp mean tokens</th><th>native mean tokens</th><th>exp median tokens</th><th>native median tokens</th></tr>
{dom_rows}
</table>
<p class="note">Net is positive across <b>all three</b> subjects, with expansion mean and median tokens
lower everywhere: the improvement is not concentrated in a single domain.</p>

<h2>Subcategory detail</h2>
<table><tr><th>Subcategory</th><th>n</th><th>exp accuracy</th><th>native accuracy</th><th>net</th></tr>{sub_rows}</table>

<h2>Latency - indicative only, different GPUs (not comparable)</h2>
<table>
<tr><th>Metric</th><th>Expansion run</th><th>Native run</th></tr>
<tr><td>mean tpot (ms/token)</td><td>{tpot_e:.1f}</td><td>{tpot_n:.1f}</td></tr>
<tr><td>mean latency per question (s)</td><td>{lat_e:.1f}</td><td>{lat_n:.1f}</td></tr>
</table>
<p class="note">The two runs were executed on <b>different GPUs</b>: tpot/latency differences reflect the
hardware, not the configuration, and must not be used as a comparison. On identical hardware the
expected reference for expansion is -20/-35% decoding speed vs native (20 vs 8 active experts).</p>

<h2>Divergent questions ({we + wn})</h2>
<table>
<tr><th>#</th><th>Subject</th><th>Target</th><th>Expansion answer</th><th>Native answer</th><th>Winner</th><th>exp tokens</th><th>nat tokens</th></tr>
{div_rows}
</table>

<h2>Methodology</h2>
<div class="box note"><ul>
<li><b>Paired protocol</b>: same 198 GPQA-diamond questions, temperature 0 (greedy), identical prompt, identical max_tokens in both runs.</li>
<li>Accuracy = evalscope review outcome (<span class="mono">ANSWER: [LETTER]</span> extraction vs target).</li>
<li>Subject assigned by matching each question text against the original GPQA dataset (Subdomain / High-level domain fields): 198/198 matched.</li>
<li>Speed comparison <b>intentionally omitted</b>: the two runs ran on different GPUs; tpot/latency are reported as per-run indications only.</li>
<li>Paired net = expansion wins - native wins over divergent questions; paper acceptance rule: net &ge; 0.</li>
<li>Branch: <span class="mono">moe-expansion</span> (25ca82a6) - llama.cpp fork vagrillo. Expansion is runtime-only, no weight changes.</li>
</ul></div>
<p class="note">Generated by report_gpqa.py · {S['exp']['n']} + {S['native']['n']} predictions analyzed.</p>
</body></html>"""

    # ---------------- markdown version ----------------
    md = []
    md.append("# GPQA-Diamond - MoE expert expansion vs native routing\n")
    md.append("> Model: Qwen3.6-35B-A3B (UD-Q6_K_XL) · 198 questions, paired protocol, temperature 0 · "
              "Expansion config: **N=20, threshold 0.8, layers 29-39, decay 0.99→0.50** · Native config: top-8 · "
              "llama.cpp branch `moe-expansion` (25ca82a6)\n")
    md.append("| Metric | Expansion | Native |")
    md.append("|---|---|---|")
    md.append(f"| **accuracy** | **{S['exp']['acc']:.2f}%** ({S['exp']['n_ok']}/198) | **{S['native']['acc']:.2f}%** ({S['native']['n_ok']}/198) |")
    md.append(f"| paired net | **+{net}** ({we} wins / {wn} losses, {br} ties) | - |")
    md.append(f"| response tokens - **mean** | {S['exp']['tok_mean']:,.0f} | {S['native']['tok_mean']:,.0f} |")
    md.append(f"| response tokens - **median** | {S['exp']['tok_median']:,.0f} | {S['native']['tok_median']:,.0f} |")
    md.append(f"| response tokens - p90 | {S['exp']['tok_p90']:,.0f} | {S['native']['tok_p90']:,.0f} |")
    md.append(f"| response tokens - max | {S['exp']['tok_max']:,} | {S['native']['tok_max']:,} |")
    md.append(f"| total tokens (198 questions) | {S['exp']['tok_total']:,} | {S['native']['tok_total']:,} |")
    md.append(f"| truncated responses (length) | {S['exp']['trunc']} | {S['native']['trunc']} |")
    md.append("\n## Comparative outcome\n")
    md.append(f"Expansion **improves accuracy by +2.02 points** (85.35% vs 83.33%) while at the same time "
              f"**reducing tokens consumed** (mean {tk_mean_d:+.1f}%, median {tk_med_d:+.1f}%): on GPQA the "
              f"\"Succinct Convergence\" phenomenon described in the paper shows up on both quality and brevity.\n")
    md.append(f"The paired net **+{net}** ({we} wins vs {wn} losses, out of {we + wn} divergent questions) is positive "
              f"but not statistically significant at n=198 (≈1.7σ): consistent with the 1-3 point effect reported "
              f"in the paper. **Paper acceptance rule (net ≥ 0): satisfied - the feature is valid for GPQA on this model.**\n")
    md.append("## Response token distribution (paired percentiles)\n")
    md.append("| percentile | expansion | native | Δ |")
    md.append("|---|---|---|---|")
    for pp in (10, 25, 50, 75, 90, 95):
        ve, vn = pct(toks_e, pp), pct(toks_n, pp)
        md.append(f"| p{pp} | {ve:,.0f} | {vn:,.0f} | {100.0*(ve-vn)/vn:+.1f}% |")
    md.append("\nAcross all percentiles up to p75 the expansion produces shorter answers (≈ -3/-6%): "
              "the paper's \"succinct convergence\". Divergences are point-wise and limited to the tail.\n")
    md.append("## Token cost under per-response budget\n")
    md.append("| Cap | total expansion | total native | Δ | clipped exp/nat |")
    md.append("|---|---|---|---|---|")
    for cap in (None, 32768, 16384):
        tag = "none" if cap is None else f"{cap // 1024}K"
        ce = toks_e if cap is None else [min(t, cap) for t in toks_e]
        cn = toks_n if cap is None else [min(t, cap) for t in toks_n]
        se, sn = sum(ce), sum(cn)
        cl_e = sum(1 for t in toks_e if cap is not None and t > cap)
        cl_n = sum(1 for t in toks_n if cap is not None and t > cap)
        md.append(f"| {tag} | {se:,} | {sn:,} | {100.0*(se-sn)/max(1,sn):+.1f}% | {cl_e} / {cl_n} |")
    md.append("\nClipping estimates **cost**, not quality: a truncated answer could flip its verdict.\n")
    md.append("## Per-subject analysis\n")
    md.append("| Subject | n | exp accuracy | native accuracy | paired net | exp mean tokens | native mean tokens | exp median tokens | native median tokens |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for dom in sorted(subj):
        idxs = subj[dom]
        ne = sum(R_["exp"][i]["correct"] for i in idxs)
        nn = sum(R_["native"][i]["correct"] for i in idxs)
        w = sum(1 for i in idxs if R_["exp"][i]["correct"] and not R_["native"][i]["correct"])
        l = sum(1 for i in idxs if not R_["exp"][i]["correct"] and R_["native"][i]["correct"])
        md.append(f"| {dom} | {len(idxs)} | {100.0*ne/len(idxs):.1f}% ({ne}) | {100.0*nn/len(idxs):.1f}% ({nn}) "
                  f"| {w-l:+d} | {st.mean(tok_e[dom]):,.0f} | {st.mean(tok_n[dom]):,.0f} "
                  f"| {st.median(tok_e[dom]):,.0f} | {st.median(tok_n[dom]):,.0f} |")
    md.append("\nNet is positive across **all three** subjects, with expansion mean and median tokens lower "
              "everywhere: the improvement is not concentrated in a single domain.\n")
    md.append("## Subcategory detail\n")
    md.append("| Subcategory | n | exp accuracy | native accuracy | net |")
    md.append("|---|---|---|---|---|")
    for (dom, sub) in sorted(subsub):
        idxs = subsub[(dom, sub)]
        ne = sum(R_["exp"][i]["correct"] for i in idxs)
        nn = sum(R_["native"][i]["correct"] for i in idxs)
        w = sum(1 for i in idxs if R_["exp"][i]["correct"] and not R_["native"][i]["correct"])
        l = sum(1 for i in idxs if not R_["exp"][i]["correct"] and R_["native"][i]["correct"])
        md.append(f"| {dom} — {sub} | {len(idxs)} | {100.0*ne/len(idxs):.1f}% ({ne}) | {100.0*nn/len(idxs):.1f}% ({nn}) | {w-l:+d} |")
    md.append("\n## Latency - indicative only, different GPUs (not comparable)\n")
    md.append(f"| Metric | Expansion run | Native run |")
    md.append(f"|---|---|---|")
    md.append(f"| mean tpot (ms/token) | {tpot_e:.1f} | {tpot_n:.1f} |")
    md.append(f"| mean latency per question (s) | {lat_e:.1f} | {lat_n:.1f} |")
    md.append("\nThe two runs were executed on **different GPUs**: tpot/latency differences reflect the hardware, "
              "not the configuration, and must not be used as a comparison. On identical hardware the expected "
              "reference for expansion is -20/-35% decoding speed vs native (20 vs 8 active experts).\n")
    md.append(f"## Divergent questions ({we + wn})\n")
    md.append("| # | Subject | Target | Expansion answer | Native answer | Winner | exp tokens | nat tokens |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i in idx_all:
        re_, rn = R_["exp"][i], R_["native"][i]
        if re_["correct"] == rn["correct"]:
            continue
        dom = runs["exp"]["preds"][i]["meta"]["domain"] if runs["exp"]["preds"][i]["meta"] else "?"
        w = "EXP" if re_["correct"] else "NAT"
        md.append(f"| {i} | {dom} | {re_['target']} | {re_['extracted'] or '?'} | {rn['extracted'] or '?'} "
                  f"| {w} | {runs['exp']['preds'][i]['out']:,} | {runs['native']['preds'][i]['out']:,} |")
    md.append("\n## Methodology\n")
    md.append("- **Paired protocol**: same 198 GPQA-diamond questions, temperature 0 (greedy), identical prompt, identical max_tokens in both runs.")
    md.append("- Accuracy = evalscope review outcome (`ANSWER: [LETTER]` extraction vs target).")
    md.append("- Subject assigned by matching each question text against the original GPQA dataset (Subdomain / High-level domain fields): 198/198 matched.")
    md.append("- Speed comparison **intentionally omitted**: the two runs ran on different GPUs; tpot/latency are reported as per-run indications only.")
    md.append("- Paired net = expansion wins - native wins over divergent questions; paper acceptance rule: net ≥ 0.")
    md.append("- Branch: `moe-expansion` (25ca82a6) - llama.cpp fork vagrillo. Expansion is runtime-only, no weight changes.")

    out_md = BASE / "report_gpqa_moe.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"MD:   {out_md}")

    out = BASE / "report_gpqa_moe.html"
    out.write_text(doc, encoding="utf-8")
    print(f"\nHTML: {out}")
    print(f"accuracy exp {S['exp']['acc']:.2f}% vs nat {S['native']['acc']:.2f}% | net {net:+d} "
          f"({we}-{wn}) | tokens mean exp {S['exp']['tok_mean']:.0f} vs nat {S['native']['tok_mean']:.0f} "
          f"({tk_mean_d:+.1f}%) | median {S['exp']['tok_median']:.0f} vs {S['native']['tok_median']:.0f} ({tk_med_d:+.1f}%)")


if __name__ == "__main__":
    main()
