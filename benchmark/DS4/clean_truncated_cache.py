#!/usr/bin/env python3
# Ripulisce la cache evalscope dalle risposte troncate: rimuove le righe delle
# domande troncate da predictions e reviews (incluso le copie rerun-*), cosi'
# che al rilancio con --use-cache evalscope le consideri mai eseguite e le rifaccia.
#
# uso:
#   python3 clean_truncated_cache.py ~/gpqa-ds4-exp --cap 8192 --dry-run   # anteprima
#   python3 clean_truncated_cache.py ~/gpqa-ds4-exp --cap 8192             # esegui
#
# dopo la pulizia, rilancia l'eval (server attivo con la stessa config):
#   evalscope eval --model DeepSeek-V4-Flash-0731-exp --eval-type openai_api \
#       --api-url http://127.0.0.1:9080/v1 --datasets gpqa_diamond \
#       --generation-config '{"temperature":0,"max_tokens":32768,"timeout":3600,"stream":true}' \
#       --eval-batch-size 1 --work-dir ~/gpqa-ds4-exp --use-cache ~/gpqa-ds4-exp --rerun-review
import argparse
import json
import re
import shutil
from pathlib import Path


def out_tokens_of(line):
    """output_tokens dal record (model_output dict o stringa)"""
    d = json.loads(line)
    mo = d.get("model_output")
    if isinstance(mo, dict):
        return (mo.get("usage") or {}).get("output_tokens"), int(d.get("index", -1))
    m = re.search(r"'output_tokens':\s*(\d+)", mo or "")
    return (int(m.group(1)) if m else None), int(d.get("index", -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--cap", type=int, default=8192, help="soglia troncamento (default 8192)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.workdir).expanduser().resolve()
    pred_files = list((root / "predictions").rglob("gpqa_diamond_default.jsonl"))
    rev_files = list((root / "reviews").rglob("gpqa_diamond_default.jsonl*"))
    assert pred_files, f"nessun predictions trovato in {root}"
    assert rev_files, f"nessun reviews trovato in {root}"

    # 1. trova gli indici delle troncate dal predictions
    trunc_idx, n_tot = set(), 0
    for pf in pred_files:
        for line in open(pf, encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            n_tot += 1
            tok, idx = out_tokens_of(line)
            if tok is not None and tok >= args.cap:
                trunc_idx.add(idx)
    print(f"record totali: {n_tot} | troncate (>= {args.cap} token): {len(trunc_idx)} "
          f"-> indici {sorted(trunc_idx)}")
    if not trunc_idx:
        print("niente da pulire")
        return

    # 2. rimuovi quelle righe da predictions e da TUTTI i reviews (base + rerun-*)
    removed_total = 0
    for f in pred_files + rev_files:
        lines = [l for l in open(f, encoding="utf-8", errors="replace") if l.strip()]
        kept, removed = [], 0
        for l in lines:
            try:
                d = json.loads(l)
                idx = int(d.get("index", -1))
            except Exception:
                kept.append(l)
                continue
            if idx in trunc_idx:
                removed += 1
            else:
                kept.append(l)
        if removed == 0:
            continue
        print(f"  {f.name}: rimuovo {removed} di {len(lines)} record "
              f"({f.parent.parent.name}/{f.parent.name})")
        removed_total += removed
        if not args.dry_run:
            # il backup NON deve stare accanto al jsonl: evalscope fa glob
            # su gpqa_diamond_default.jsonl* e leggerebbe anche i .bak
            bdir = root / "backup_before_clean"
            bdir.mkdir(exist_ok=True)
            bak = bdir / (f.parent.name + "__" + f.name)
            shutil.copy2(f, bak)
            f.write_text("".join(kept), encoding="utf-8")

    print(f"\ntotale record rimossi: {removed_total}")
    if args.dry_run:
        print("(dry-run: nessuna modifica scritta)")
    else:
        print(f"backup in: {root / 'backup_before_clean'}")
        print("\nora rilancia l'eval con --use-cache: evalscope rifara' solo le domande "
              "rimosse (oltre a quelle mai eseguite).")
        print("se l'identity check protesta, aggiungi --rerun-review (gia' nel tuo bench.sh).")


if __name__ == "__main__":
    main()
