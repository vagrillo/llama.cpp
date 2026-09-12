#!/usr/bin/env python3
# RUN1209: analisi comparativa GPQA-diamond — 5 configurazioni:
#   DS4-Flash (IQ2_M): native top-6 vs expansion N=12/T=0.8/L28-42/decay0.10
#   Qwen3.6-35B (Q8_0): native top-8 vs expansion N=16/T=0.8/L25-39/decay0.50
#   Qwen3.8-27B Q8: riferimento nativo
# Genera RUN1209_comparison.md + RUN1209_report.html (paper appendix, EN)
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
    urllib.request.urlretrieve(DATASET_URL, DATASET)


def norm(s, limit=6000):
    return " ".join(str(s).split()).lower()[:limit]


def load_dataset():
    gpqa = {}
    for line in open(DATASET, encoding="utf-8"):
        d = json.loads(line)
        gpqa[norm(d["Question"], 120)] = d.get("High-level domain") or "?"
    return gpqa


def load_run(root, gpqa):
    P, R = {}, {}
    pf = list((root / "predictions").rglob("gpqa_diamond_default.jsonl"))[0]
    for line in open(pf, encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        d = json.loads(line)
        mo = d.get("model_output")
        prompt = ""
        for m in d.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    prompt = c
        key = norm(prompt)
        meta = None
        for q, v in gpqa.items():
            if q and q in key:
                meta = v
                break
        if isinstance(mo, dict):
            u = mo.get("usage") or {}
            ch = (mo.get("choices") or [{}])[0]
            tp = (mo.get("perf_metrics") or {}).get("tpot")
            rec = {"out": u.get("output_tokens") or 0,
                   "stop": ch.get("stop_reason", "?"),
                   "tpot": tp * 1000 if tp else None}
        else:
            s = mo or ""

            def rx(pat, cast=float):
                m = re.search(pat, s)
                return cast(m.group(1)) if m else None
            tp = rx(r"'tpot':\s*([0-9.]+)")
            rec = {"out": rx(r"'output_tokens':\s*(\d+)", int) or 0,
                   "stop": (re.search(r"'stop_reason':\s*'([^']+)'", s) or [None, "?"])[1],
                   "tpot": tp * 1000 if tp else None}
        rec["domain"] = meta or "?"
        P[int(d["index"])] = rec
    rf = list((root / "reviews").rglob("gpqa_diamond_default.jsonl"))[0]
    for line in open(rf, encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        d = json.loads(line)
        sc = (d.get("sample_score") or {}).get("score", {})
        val = (sc.get("value") or {}).get("accuracy")
        R[int(d["index"])] = {"correct": bool(val and val >= 0.5),
                              "extracted": sc.get("extracted_prediction"),
                              "target": d.get("target")}
    return P, R


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p / 100))]


def load_all(gpqa):
    runs = {}
    specs = {
        "ds4_exp": BASE / "RunDS41209/root/gpqa-ds4-exp",
        "ds4_native": BASE / "RunDS41209/root/gpqa-ds4-native",
        "q35_exp": BASE / "RunQ351209/root/gpqa-exp",
        "q35_native": BASE / "RunQ351209/root/gpqa-native",
        "q27b": BASE / "RunQ351209/root/gpqa-27BQ8",
    }
    for k, d in specs.items():
        runs[k] = load_run(d, gpqa)
    return runs


def accuracy(runs, key, idxs):
    R = runs[key][1]
    ks = [i for i in idxs if i in R]
    return 100.0 * sum(R[i]["correct"] for i in ks) / max(1, len(ks)), sum(R[i]["correct"] for i in ks), len(ks)


def main():
    gpqa = load_dataset()
    runs = load_all(gpqa)

    common = set(runs["ds4_exp"][0]) & set(runs["ds4_native"][0]) & \
             set(runs["ds4_exp"][1]) & set(runs["ds4_native"][1]) & \
             set(runs["q35_exp"][0]) & set(runs["q35_native"][0]) & \
             set(runs["q35_exp"][1]) & set(runs["q35_native"][1]) & \
             set(runs["q27b"][0]) & set(runs["q27b"][1])
    common = sorted(common)
    n = len(common)
    print(f"domande comuni a tutti i run: {n}")

    # per-config stats
    stats = {}
    for key in ("ds4_exp", "ds4_native", "q35_exp", "q35_native", "q27b"):
        P, R = runs[key]
        ks = [i for i in common if i in R and i in P]
        toks = [P[i]["out"] for i in ks]
        corr = sum(R[i]["correct"] for i in ks)
        stats[key] = {"n": len(ks), "acc": 100.0 * corr / max(1, len(ks)), "n_ok": corr,
                      "tok_mean": st.mean(toks), "tok_median": st.median(toks),
                      "tok_p90": pct(toks, 90), "tok_max": max(toks), "tok_total": sum(toks),
                      "trunc": sum(1 for i in ks if P[i]["stop"] not in ("stop", "?"))}

    # paired espansione vs nativo per modello
    def paired(pa, pb, idxs):
        we_ = sum(1 for i in idxs if runs[pa][1][i]["correct"] and not runs[pb][1][i]["correct"])
        wn_ = sum(1 for i in idxs if runs[pb][1][i]["correct"] and not runs[pa][1][i]["correct"])
        bo = sum(1 for i in idxs if runs[pa][1][i]["correct"] and runs[pb][1][i]["correct"])
        no = sum(1 for i in idxs if not runs[pa][1][i]["correct"] and not runs[pb][1][i]["correct"])
        return we_, wn_, bo, no

    d_p = paired("ds4_exp", "ds4_native", common)
    q_p = paired("q35_exp", "q35_native", common)

    # per materia
    subj = defaultdict(list)
    for i in common:
        m = runs["ds4_exp"][0][i]["domain"]
        subj[m].append(i)

    # cross-model: q35_exp vs q27b
    x_common = sorted(set(runs["q35_exp"][1]) & set(runs["q27b"][1]))
    x_q = 100.0 * sum(runs["q35_exp"][1][i]["correct"] for i in x_common) / max(1, len(x_common))
    x_27 = 100.0 * sum(runs["q27b"][1][i]["correct"] for i in x_common) / max(1, len(x_common))

    out = {"n": n, "stats": stats, "paired": {"ds4": d_p, "q35": q_p},
           "subj": {dom: {"n": len(idxs),
                          "acc": {k: 100.0 * sum(runs[k][1][i]["correct"] for i in idxs) / len(idxs)
                                  for k in stats},
                          "net_ds4": sum(1 for i in idxs if runs["ds4_exp"][1][i]["correct"] and not runs["ds4_native"][1][i]["correct"])
                                     - sum(1 for i in idxs if runs["ds4_native"][1][i]["correct"] and not runs["ds4_exp"][1][i]["correct"]),
                          "net_q35": sum(1 for i in idxs if runs["q35_exp"][1][i]["correct"] and not runs["q35_native"][1][i]["correct"])
                                     - sum(1 for i in idxs if runs["q35_native"][1][i]["correct"] and not runs["q35_exp"][1][i]["correct"])}
                     for dom, idxs in subj.items()},
           "x_q35": x_q, "x_27": x_27}
    json.dump(out, open(BASE / "RUN1209_stats.json", "w"), indent=1, default=str)
    print(json.dumps(out["subj"], indent=1, default=str))
    print("salvato RUN1209_stats.json")


if __name__ == "__main__":
    main()
