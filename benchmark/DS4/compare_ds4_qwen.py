#!/usr/bin/env python3
# Confronto preliminare: DeepSeek-V4-Flash-0731 (Q2, espanso) vs Qwen3.6-35B-A3B
# (Q6_K_XL, espanso) sulle stesse domande GPQA-diamond.
# NB: modelli E quant diversi -> il confronto di accuracy e' cross-model, non
#     misura l'effetto dell'espansione (che richiede il paired nativo di DS4).
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
DATASET = Path("/tmp/gpqa_train.jsonl")


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


def load_run(pred_file, rev_files):
    gpqa = load_dataset()
    preds, revs = {}, {}
    for line in open(pred_file, encoding="utf-8", errors="replace"):
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
        key = norm(qtext)
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
        meta = None
        for k, v in gpqa.items():
            if k in key:
                meta = v
                break
        rec["meta"] = meta
        rec["qkey"] = key
        preds[key] = rec
    for rf in rev_files:
        for line in open(rf, encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            d = json.loads(line)
            sc = (d.get("sample_score") or {}).get("score", {})
            val = (sc.get("value") or {}).get("accuracy")
            q = d.get("messages") or []
            qtext = ""
            for m in q:
                if isinstance(m, dict) and m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, list):
                        c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                    if isinstance(c, str):
                        qtext = c
            revs[norm(qtext)] = {"correct": bool(val and val >= 0.5),
                                      "extracted": sc.get("extracted_prediction"),
                                      "target": d.get("target")}
    return preds, revs


REPO_BENCH = Path("/mnt/c/AI/src/llamacpp/repo/benchmark/GPQA")
BASE_PRED = REPO_BENCH / "gpqa-exp/predictions/Qwen3.6-35B-A3B-exp/gpqa_diamond_default.jsonl"
BASE_REV = sorted((REPO_BENCH / "gpqa-exp/reviews/Qwen3.6-35B-A3B-exp").glob("gpqa_diamond_default.jsonl"))
DS4_PRED = BASE / "gpqa-ds4-exp/predictions/DeepSeek-V4-Flash-0731-exp/gpqa_diamond_default.jsonl"
DS4_REV = sorted((BASE / "gpqa-ds4-exp/reviews/DeepSeek-V4-Flash-0731-exp").glob("gpqa_diamond_default.jsonl*"))

qw_p, qw_r = load_run(BASE_PRED, BASE_REV)
ds_p, ds_r = load_run(DS4_PRED, DS4_REV)
common = sorted(set(qw_p) & set(ds_p) & set(qw_r) & set(ds_r))
print(f"Qwen-exp: {len(qw_p)} predizioni, {len(qw_r)} review")
print(f"DS4-exp : {len(ds_p)} predizioni, {len(ds_r)} review")
print(f"domande in comune (match per testo): {len(common)}")

if not common:
    raise SystemExit(1)


def stats(P, R, idxs, label):
    toks = [P[i]["out"] for i in idxs if P[i]["out"] and P[i]["out"] > 0]
    corr = [R[i]["correct"] for i in idxs if i in R]
    acc = 100.0 * sum(corr) / max(1, len(corr))
    print(f"\n=== {label} ===")
    print(f"  accuracy      : {acc:.1f}% ({sum(corr)}/{len(corr)})")
    if toks:
        print(f"  token risposta: media {st.mean(toks):,.0f} | mediana {st.median(toks):,.0f} | "
              f"p90 {pct(toks,90):,.0f} | max {max(toks):,}")
    trunc = sum(1 for i in idxs if P[i]["stop"] not in ("stop",))
    print(f"  troncate      : {trunc}")
    tp = [P[i]["tpot"] for i in idxs if P[i]["tpot"]]
    if tp:
        print(f"  tpot (indicativo, HW diversi): {st.mean(tp):.1f} ms/token")
    return acc, toks


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p / 100))]


a_q, t_q = stats(qw_p, qw_r, common, "QWEN3.6-35B-A3B (Q6_K_XL) ESPANSO N=20 T=0.8")
a_d, t_d = stats(ds_p, ds_r, common, "DEEPSEEK-V4-FLASH-0731 (Q2) ESPANSO (config del run)")

# agreement cross-model
both = sum(1 for i in common if qw_r[i]["correct"] and ds_r[i]["correct"])
none = sum(1 for i in common if not qw_r[i]["correct"] and not ds_r[i]["correct"])
q_only = sum(1 for i in common if qw_r[i]["correct"] and not ds_r[i]["correct"])
d_only = sum(1 for i in common if not qw_r[i]["correct"] and ds_r[i]["correct"])
print(f"\n=== ACCORDO CROSS-MODEL sulle {len(common)} domande ===")
print(f"entrambe corrette : {both}")
print(f"solo qwen         : {q_only}")
print(f"solo ds4          : {d_only}")
print(f"entrambe sbagliate: {none}")

# ---- confronto escludendo le risposte DS4 troncate (stop_reason=length) ----
complete = [i for i in common if ds_p[i]["stop"] == "stop"]
truncset = [i for i in common if ds_p[i]["stop"] != "stop"]
print(f"\n=== SOLO RISPOSTE COMPLETE (DS4, stop!=length): {len(complete)} domande "
      f"({len(truncset)} troncate escluse) ===")
a_q2, _ = stats(qw_p, qw_r, complete, "QWEN espanso (sulle domande complete DS4)")
a_d2, t_d2 = stats(ds_p, ds_r, complete, "DS4 espanso (senza troncate)")
b2 = sum(1 for i in complete if qw_r[i]["correct"] and ds_r[i]["correct"])
w2 = sum(1 for i in complete if qw_r[i]["correct"] and not ds_r[i]["correct"])
l2 = sum(1 for i in complete if not qw_r[i]["correct"] and ds_r[i]["correct"])
n2 = sum(1 for i in complete if not qw_r[i]["correct"] and not ds_r[i]["correct"])
print(f"  paired: entrambe ok {b2} | vince qwen {w2} | vince ds4 {l2} | entrambe ko {n2}")
print(f"  tokens ds4 complete: media {st.mean(t_d2):,.0f} | mediana {st.median(t_d2):,.0f}")
print(f"\n=== SOLO LE TRONCATE (DS4 stop=length): {len(truncset)} domande ===")
if truncset:
    ad = 100.0 * sum(ds_r[i]["correct"] for i in truncset) / len(truncset)
    aq = 100.0 * sum(qw_r[i]["correct"] for i in truncset) / len(truncset)
    tq = [qw_p[i]["out"] for i in truncset if qw_p[i]["out"]]
    print(f"  ds4 accuracy (giudicate senza ANSWER finale): {ad:.1f}%")
    print(f"  qwen accuracy sulle stesse domande          : {aq:.1f}%")
    if tq:
        print(f"  qwen token medi su quelle domande           : {st.mean(tq):,.0f}")

# per materia
subj = defaultdict(list)
for i in common:
    m = qw_p[i]["meta"]
    if m:
        subj[m["domain"]].append(i)
print("\n=== PER MATERIA (accuracy ds4-exp | qwen-exp | token medi ds4 | qwen) ===")
for dom in sorted(subj):
    idxs = subj[dom]
    ad = 100.0 * sum(ds_r[i]["correct"] for i in idxs) / len(idxs)
    aq = 100.0 * sum(qw_r[i]["correct"] for i in idxs) / len(idxs)
    td = [ds_p[i]["out"] for i in idxs if ds_p[i]["out"]]
    tq = [qw_p[i]["out"] for i in idxs if qw_p[i]["out"]]
    print(f"  {dom:10s} n={len(idxs):3d}  ds4={ad:5.1f}%  qwen={aq:5.1f}%  "
          f"tok ds4={st.mean(td):6.0f}  tok qwen={st.mean(tq):6.0f}")
