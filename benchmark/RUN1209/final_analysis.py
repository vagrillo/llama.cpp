#!/usr/bin/env python3
# Final comparative analysis over RUN1209 (5 runs, 198 GPQA-diamond questions).
# Computes: accuracy, per-subject accuracy, paired/common-question matrices,
# token-bucket accuracy (reasoning length classes), loop detection, perf stats.
# Writes final_analysis.json next to this script.

import json
import re
import zlib
import statistics as st
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
DATASET = Path("/tmp/gpqa_train.jsonl")

RUNS = {
    "q35_exp": {
        "pred": BASE / "RunQ351209/root/gpqa-exp/predictions/Qwen3.6-35B-A3B-exp/gpqa_diamond_default.jsonl",
        "rev": BASE / "RunQ351209/root/gpqa-exp/reviews/Qwen3.6-35B-A3B-exp/gpqa_diamond_default.jsonl",
    },
    "q35_native": {
        "pred": BASE / "RunQ351209/root/gpqa-native/predictions/Qwen3.6-35B-A3B-native/gpqa_diamond_default.jsonl",
        "rev": BASE / "RunQ351209/root/gpqa-native/reviews/Qwen3.6-35B-A3B-native/gpqa_diamond_default.jsonl",
    },
    "q27b": {
        "pred": BASE / "RunQ351209/root/gpqa-27BQ8/predictions/Qwen3.8-27BQ8/gpqa_diamond_default.jsonl",
        "rev": BASE / "RunQ351209/root/gpqa-27BQ8/reviews/Qwen3.8-27BQ8/gpqa_diamond_default.jsonl",
    },
    "ds4_exp": {
        "pred": BASE / "RunDS41209/root/gpqa-ds4-exp/predictions/DeepSeek-V4-Flash-0731-exp/gpqa_diamond_default.jsonl",
        "rev": BASE / "RunDS41209/root/gpqa-ds4-exp/reviews/DeepSeek-V4-Flash-0731-exp/gpqa_diamond_default.jsonl",
    },
    "ds4_native": {
        "pred": BASE / "RunDS41209/root/gpqa-ds4-native/predictions/DeepSeek-V4-Flash-0731-native/gpqa_diamond_default.jsonl",
        "rev": BASE / "RunDS41209/root/gpqa-ds4-native/reviews/DeepSeek-V4-Flash-0731-native/gpqa_diamond_default.jsonl",
    },
}


def norm(s, limit=None):
    s = " ".join(str(s).split()).lower()
    return s[:limit] if limit else s


def load_dataset():
    gpqa = []
    for line in open(DATASET, encoding="utf-8"):
        d = json.loads(line)
        gpqa.append({
            "key": norm(d["Question"], 120),
            "domain": d.get("High-level domain") or "?",
            "subdomain": d.get("Subdomain") or "?",
        })
    return gpqa


def user_text(rec):
    qtext = ""
    for m in rec.get("messages") or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
            if isinstance(c, str) and len(c) > len(qtext):
                qtext = c
    return qtext


def reasoning_text(mo):
    """concatenate all reasoning blocks"""
    out = []
    try:
        for ch in mo.get("choices") or []:
            for part in ((ch.get("message") or {}).get("content") or []):
                if isinstance(part, dict) and part.get("type") == "reasoning":
                    out.append(part.get("reasoning") or "")
    except AttributeError:
        pass
    return "\n".join(out)


def final_text(mo):
    out = []
    try:
        for ch in mo.get("choices") or []:
            for part in ((ch.get("message") or {}).get("content") or []):
                if isinstance(part, dict) and part.get("type") == "text":
                    out.append(part.get("text") or "")
    except AttributeError:
        pass
    return "\n".join(out)


def loop_metrics(reasoning):
    """loop/repetition signals from reasoning text"""
    if not reasoning or len(reasoning) < 800:
        return {"zratio": None, "maxline": 0, "loop": False}
    zratio = len(zlib.compress(reasoning.encode("utf-8"), 9)) / len(reasoning.encode("utf-8"))
    lines = [l.strip() for l in reasoning.splitlines() if len(l.strip()) > 25]
    cnt = defaultdict(int)
    for l in lines:
        cnt[l] += 1
    maxline = max(cnt.values()) if cnt else 0
    loop = (zratio < 0.30 and len(reasoning) > 6000) or maxline >= 5
    return {"zratio": round(zratio, 3), "maxline": maxline, "loop": loop}


def load_run(runcfg, gpqa):
    preds, revs = {}, {}
    for i, line in enumerate(open(runcfg["pred"], encoding="utf-8", errors="replace")):
        if not line.strip():
            continue
        d = json.loads(line)
        idx = d["index"]
        mo = d.get("model_output") or {}
        u = mo.get("usage") or {}
        pm = mo.get("perf_metrics") or {}
        ch = (mo.get("choices") or [{}])[0]
        qtext = user_text(d)
        key = norm(qtext)
        meta = next((g for g in gpqa if g["key"] in key), None)
        tpot = pm.get("tpot")
        rec = {
            "idx": idx,
            "domain": meta["domain"] if meta else "?",
            "subdomain": meta["subdomain"] if meta else "?",
            "qkey": key,
            "out": u.get("output_tokens"),
            "inp": u.get("input_tokens"),
            "lat": pm.get("latency") or mo.get("time"),
            "ttft": pm.get("ttft"),
            "tpot_ms": tpot * 1000 if tpot else None,
            "stop": ch.get("stop_reason", "?"),
            "reason": reasoning_text(mo),
            "final": final_text(mo),
        }
        rec["loop"] = loop_metrics(rec["reason"])
        preds[idx] = rec
    for line in open(runcfg["rev"], encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        d = json.loads(line)
        idx = d["index"]
        sc = (d.get("sample_score") or {}).get("score", {})
        val = (sc.get("value") or {}).get("accuracy")
        revs[idx] = {
            "correct": bool(val and val >= 0.5),
            "extracted": sc.get("extracted_prediction"),
            "target": d.get("target"),
        }
    return preds, revs


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p / 100))]


def agg(nums):
    nums = [x for x in nums if x is not None and x > 0]
    if not nums:
        return None
    return {
        "mean": round(st.mean(nums), 1),
        "median": round(st.median(nums), 1),
        "p90": pct(nums, 90),
        "max": max(nums),
        "min": min(nums),
        "total": sum(nums),
    }


BUCKETS = [
    ("brief (<2k)", 0, 2000),
    ("normal (2-5k)", 2000, 5000),
    ("medium (5-10k)", 5000, 10000),
    ("long (10-20k)", 10000, 20000),
    ("very long (>20k)", 20000, 10**9),
]


def main():
    gpqa = load_dataset()
    print(f"dataset rows: {len(gpqa)}")
    runs = {}
    for name, cfg in RUNS.items():
        P, R = load_run(cfg, gpqa)
        assert len(P) == 198 and len(R) == 198, f"{name}: {len(P)} preds, {len(R)} revs"
        runs[name] = (P, R)
        nodom = sum(1 for p in P.values() if p["domain"] == "?")
        print(f"{name}: 198 ok, unmatched domain: {nodom}")

    summary = {}
    for name, (P, R) in runs.items():
        accs = [R[i]["correct"] for i in range(198)]
        toks = [P[i]["out"] for i in range(198)]
        lats = [P[i]["lat"] for i in range(198)]
        tps = [P[i]["out"] / P[i]["lat"] for i in range(198) if P[i]["lat"] and P[i]["out"]]
        tpots = [P[i]["tpot_ms"] for i in range(198) if P[i]["tpot_ms"]]
        ttfts = [P[i]["ttft"] for i in range(198) if P[i]["ttft"]]
        trunc = sum(1 for i in range(198) if P[i]["stop"] != "stop")
        loops = sum(1 for i in range(198) if P[i]["loop"]["loop"])
        buckets = []
        for label, lo, hi in BUCKETS:
            idxs = [i for i in range(198) if lo <= (P[i]["out"] or 0) < hi]
            b_acc = 100.0 * sum(R[i]["correct"] for i in idxs) / len(idxs) if idxs else None
            b_tok = st.mean([P[i]["out"] for i in idxs]) if idxs else None
            buckets.append({"bucket": label, "n": len(idxs), "acc": round(b_acc, 2) if b_acc is not None else None,
                            "mean_tok": round(b_tok, 1) if b_tok is not None else None})
        subjects = {}
        for dom in ("Chemistry", "Physics", "Biology"):
            idxs = [i for i in range(198) if P[i]["domain"] == dom]
            subjects[dom] = {
                "n": len(idxs),
                "acc": round(100.0 * sum(R[i]["correct"] for i in idxs) / len(idxs), 2),
                "mean_tok": round(st.mean([P[i]["out"] for i in idxs]), 1),
                "mean_lat": round(st.mean([P[i]["lat"] for i in idxs]), 1),
            }
        summary[name] = {
            "acc": round(100.0 * sum(accs) / 198, 2),
            "n_ok": sum(accs),
            "tok": agg(toks),
            "latency_s": agg(lats),
            "tok_per_s_mean_of_q": round(st.mean(tps), 2),
            "tok_per_s_aggregate": round(sum(toks) / sum(lats), 2),
            "tpot_ms": round(st.mean(tpots), 2) if tpots else None,
            "ttft_s": round(st.mean(ttfts), 2) if ttfts else None,
            "trunc": trunc,
            "loop_suspects": loops,
            "buckets": buckets,
            "subjects": subjects,
        }

    # per-question correctness matrix + common-question analysis
    C = {name: [runs[name][1][i]["correct"] for i in range(198)] for name in runs}

    def paired(a, b):
        wa = sum(1 for i in range(198) if C[a][i] and not C[b][i])
        wb = sum(1 for i in range(198) if C[b][i] and not C[a][i])
        both = sum(1 for i in range(198) if C[a][i] and C[b][i])
        none = sum(1 for i in range(198) if not C[a][i] and not C[b][i])
        return {"a_wins": wa, "b_wins": wb, "both_right": both, "both_wrong": none, "net": wa - wb}

    paired_sets = {
        "q35_exp_vs_q35_native": paired("q35_exp", "q35_native"),
        "ds4_exp_vs_ds4_native": paired("ds4_exp", "ds4_native"),
        "q35_exp_vs_q27b": paired("q35_exp", "q27b"),
        "ds4_exp_vs_q35_exp": paired("ds4_exp", "q35_exp"),
        "ds4_exp_vs_q35_native": paired("ds4_exp", "q35_native"),
    }
    all5 = {
        "all_right": sum(1 for i in range(198) if all(C[n][i] for n in runs)),
        "all_wrong": sum(1 for i in range(198) if not any(C[n][i] for n in runs)),
    }
    # questions where all 4 MoE/expanded-and-native runs agree vs 27b etc.
    universe = []
    P0 = runs["q35_exp"][0]
    for i in range(198):
        states = {n: C[n][i] for n in runs}
        universe.append({
            "idx": i,
            "domain": P0[i]["domain"],
            "states": states,
            "n_right": sum(states.values()),
        })

    # divergent question details for the two key pairings
    def divergent(a, b, limit=40):
        out = []
        for i in range(198):
            if C[a][i] != C[b][i]:
                out.append({
                    "idx": i,
                    "domain": P0[i]["domain"],
                    "winner": a if C[a][i] else b,
                    "q": P0[i]["qkey"][:110],
                })
        return out[:limit], len(out)

    div = {}
    div["q35"], div["q35_n"] = divergent("q35_exp", "q35_native")
    div["ds4"], div["ds4_n"] = divergent("ds4_exp", "ds4_native")

    # loop suspect details
    loop_details = {}
    for name, (P, R) in runs.items():
        det = []
        for i in range(198):
            if P[i]["loop"]["loop"]:
                det.append({
                    "idx": i, "domain": P[i]["domain"], "correct": R[i]["correct"],
                    "out": P[i]["out"], "stop": P[i]["stop"],
                    "zratio": P[i]["loop"]["zratio"], "maxline": P[i]["loop"]["maxline"],
                    "q": P[i]["qkey"][:90],
                })
        loop_details[name] = det

    # matched-question comparison: restrict token/latency to subsets where the
    # two runs share the same outcome (both-correct ties = reviewer suggestion),
    # so differences reflect routing cost rather than question mix
    def restricted(a, b):
        def sub(P, idxs):
            pairs = [(P[i]["out"] or 0, P[i]["lat"]) for i in idxs if P[i]["lat"]]
            tok = [t for t, _ in pairs]
            lat = [l for _, l in pairs]
            return {
                "tok_mean": round(st.mean(tok), 1),
                "tok_median": round(st.median(tok), 1),
                "lat_mean": round(st.mean(lat), 2),
                "tok_per_s_mean": round(st.mean([t / l for t, l in pairs]), 2),
                "tok_per_s_agg": round(sum(tok) / sum(lat), 3),
            }
        out = {}
        for label, idxs in (
            ("both_correct", [i for i in range(198) if C[a][i] and C[b][i]]),
            ("same_outcome", [i for i in range(198) if C[a][i] == C[b][i]]),
        ):
            sa, sb = sub(runs[a][0], idxs), sub(runs[b][0], idxs)
            out[label] = {
                "n": len(idxs),
                a: sa, b: sb,
                "tok_delta_pct": round(100 * (sa["tok_mean"] / sb["tok_mean"] - 1), 2),
                "lat_delta_pct": round(100 * (sa["lat_mean"] / sb["lat_mean"] - 1), 2),
            }
        return out

    restricted_sets = {
        "q35_exp_vs_q35_native": restricted("q35_exp", "q35_native"),
        "ds4_exp_vs_ds4_native": restricted("ds4_exp", "ds4_native"),
    }

    out = {
        "summary": summary,
        "paired": paired_sets,
        "restricted": restricted_sets,
        "all5": all5,
        "divergent": div,
        "loop_details": loop_details,
        "universe": universe,
    }
    outpath = BASE / "final_analysis.json"
    outpath.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {outpath}")

    # quick console digest
    for name in RUNS:
        s = summary[name]
        print(f"\n== {name}: acc {s['acc']}% tok_mean {s['tok']['mean']} lat_mean {s['latency_s']['mean']}s "
              f"tps {s['tok_per_s_mean_of_q']} trunc {s['trunc']} loops {s['loop_suspects']}")
        for b in s["buckets"]:
            print(f"   {b['bucket']:18s} n={b['n']:3d} acc={b['acc']}")
        for d, v in s["subjects"].items():
            print(f"   {d:10s} n={v['n']:3d} acc={v['acc']}%")
    print("\npaired:")
    for k, v in paired_sets.items():
        print(f"  {k}: {v}")
    print(f"\nall5: {all5}")


if __name__ == "__main__":
    main()
