#!/usr/bin/env python3
# Rerun ONLY the GPQA questions where one configuration beat the other, against
# the RUNNING llama-server (must serve the EXPANSION config, --parallel 1,
# context >= prompt + --newcap).
#
# Selection (--select):
#   nat_better    native correct & expansion wrong   (default: the net-losing set)
#   exp_better    expansion correct & native wrong
#   exp_truncated expansion answers that hit the length cap (stop=length)
#   nat_truncated native answers that hit the length cap
#   all_divergent every divergent question
#
# uso:
#   python3 eval_divergent.py --exp-dir ~/gpqa-ds4-exp --nat-dir ~/gpqa-ds4-native \
#       --port 9080 --newcap 32768 --select nat_better
#
# output: live per-question results + a JSON report (--out) with the projected
# combined net after the redo.
import argparse
import json
import re
import statistics as st
import urllib.request
from pathlib import Path


def norm(s, limit=6000):
    return " ".join(str(s).split()).lower()[:limit]


def load_run(workdir: Path):
    """prompt-key -> record, from an evalscope work dir (predictions + reviews)"""
    workdir = workdir.expanduser().resolve()
    cand = sorted((workdir / "predictions").rglob("gpqa_diamond_default.jsonl")) if \
        (workdir / "predictions").exists() else []
    if not cand:
        # fallback: cerca le predictions ovunque sotto la dir data
        cand = sorted(workdir.rglob("predictions/*/gpqa_diamond_default.jsonl"))
    if not cand:
        raise SystemExit(f"ERROR: nessuna predictions trovata sotto {workdir}\n"
                         f"  verifica i work-dir reali (es. ls ~/*/predictions)")
    pf = cand[-1]
    for line in open(pf, encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        d = json.loads(line)
        mo = d.get("model_output")
        prompt, out_tokens, stop = "", 0, "?"
        for m in d.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                if isinstance(c, str):
                    prompt = c
        if isinstance(mo, dict):
            u = mo.get("usage") or {}
            ch = (mo.get("choices") or [{}])[0]
            out_tokens, stop = u.get("output_tokens") or 0, ch.get("stop_reason", "?")
        else:
            mo = mo or ""
            m = re.search(r"'output_tokens':\s*(\d+)", mo)
            out_tokens = int(m.group(1)) if m else 0
            s = re.search(r"'stop_reason':\s*'([^']+)'", mo)
            stop = s.group(1) if s else "?"
        preds[norm(prompt)] = {"prompt": prompt, "out": out_tokens, "stop": stop}
    rf_dir = workdir / "reviews"
    base = list(rf_dir.rglob("gpqa_diamond_default.jsonl"))
    rf = base[0] if base else sorted(rf_dir.rglob("gpqa_diamond_default.jsonl.rerun-*"))[-1]
    for line in open(rf, encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        d = json.loads(line)
        sc = (d.get("sample_score") or {}).get("score", {})
        val = (sc.get("value") or {}).get("accuracy")
        qtext = ""
        for m in d.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    qtext = c
        revs[norm(qtext)] = {"correct": bool(val and val >= 0.5),
                             "extracted": sc.get("extracted_prediction"),
                             "target": d.get("target")}
    return preds, revs


def extract_answer(text):
    hits = re.findall(r"ANSWER:\s*\**\s*([A-D])\b", text or "")
    return hits[-1] if hits else None


def query(base_url, prompt, max_tokens):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    content = ch["message"].get("content")
    if isinstance(content, list):
        content = "\n".join(p.get("text", "") or p.get("reasoning", "")
                            for p in content if isinstance(p, dict))
    usage = d.get("usage") or {}
    tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    stop = ch.get("stop_reason") or ch.get("finish_reason") or "?"
    return content or "", int(tokens), stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, help="evalscope work dir of the EXPANSION run")
    ap.add_argument("--nat-dir", required=True, help="evalscope work dir of the NATIVE run")
    ap.add_argument("--port", type=int, default=9080)
    ap.add_argument("--newcap", type=int, default=32768, help="absolute cap for the redo")
    ap.add_argument("--native-margin", type=float, default=1.5,
                    help="per-question budget = native tokens * this margin (capped at --newcap)")
    ap.add_argument("--select", default="nat_better",
                    choices=["nat_better", "exp_better", "exp_truncated",
                             "nat_truncated", "all_divergent"])
    ap.add_argument("--out", default="gpqa_divergent_redo.json")
    args = ap.parse_args()

    pe, re_ = load_run(Path(args.exp_dir).expanduser())
    pn, rn = load_run(Path(args.nat_dir).expanduser())
    common = sorted(set(pe) & set(pn) & set(re_) & set(rn))
    print(f"domande in comune: {len(common)} (exp {len(pe)}, nat {len(pn)})")

    sel = []
    for i in common:
        e_ok, n_ok = re_[i]["correct"], rn[i]["correct"]
        e_trunc = pe[i]["stop"] not in ("stop", "?")
        n_trunc = pn[i]["stop"] not in ("stop", "?")
        if args.select == "nat_better" and n_ok and not e_ok:
            sel.append(i)
        elif args.select == "exp_better" and e_ok and not n_ok:
            sel.append(i)
        elif args.select == "exp_truncated" and e_trunc:
            sel.append(i)
        elif args.select == "nat_truncated" and n_trunc:
            sel.append(i)
        elif args.select == "all_divergent" and e_ok != n_ok:
            sel.append(i)
    print(f"selezionate (--select {args.select}): {len(sel)}")
    if not sel:
        print("niente da rifare")
        return

    base = f"http://127.0.0.1:{args.port}"
    results, n_ok = [], 0
    for n, i in enumerate(sel, 1):
        # budget per domanda: token usati dal nativo su quella domanda * margine
        # (default +50%), mai oltre il cap assoluto --newcap
        nat_t = pn[i].get("out") or 0
        budget = min(args.newcap, max(1024, int(nat_t * args.native_margin)))
        content, tokens, stop = query(base, pe[i]["prompt"], budget)
        ans = extract_answer(content)
        tgt = re_[i]["target"]
        ok = ans is not None and tgt is not None and ans == tgt
        n_ok += ok
        results.append({"index_key": i[:60], "target": tgt, "answer": ans,
                        "correct": ok, "tokens": tokens, "stop": stop, "budget": budget,
                        "was": {"exp_correct": re_[i]["correct"],
                                "exp_tokens": pe[i]["out"], "exp_stop": pe[i]["stop"],
                                "nat_tokens": nat_t}})
        tag = "FLIP OK" if ok else ("ko (cap)" if stop == "length" else "ko")
        print(f"  [{n}/{len(sel)}] nat={nat_t} budget={budget} tokens={tokens} stop={stop} "
              f"ans={ans} target={tgt} -> {tag}")

    # proiezione: le selezionate ora corrette passano all'espanso
    cur_we = sum(1 for i in common if re_[i]["correct"] and not rn[i]["correct"])
    cur_wn = sum(1 for i in common if not re_[i]["correct"] and rn[i]["correct"])
    flips = n_ok
    new_we = cur_we + (flips if args.select in ("nat_better", "all_divergent") else 0)
    new_wn = cur_wn + (flips if args.select in ("exp_better",) else 0)
    print("\n=== RISULTATO ===")
    print(f"corrette nel redo     : {n_ok}/{len(sel)}")
    print(f"paired net attuale    : {cur_we - cur_wn:+d} (exp {cur_we} vs nat {cur_wn})")
    print(f"proiezione post-redo  : {new_we - new_wn:+d}")

    out = Path(args.out)
    out.write_text(json.dumps({"select": args.select, "n": len(sel), "results": results,
                               "projected_net": new_we - new_wn}, indent=1))
    print(f"salvato: {out}")


if __name__ == "__main__":
    main()
