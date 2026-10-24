# Per-Expert Mixed-Precision Quantization (split pools)

Quantizzazione differenziata per esperto dei modelli MoE, guidata dal ranking
d'uso raccolto a runtime: gli esperti più utilizzati di ogni layer vengono
quantizzati a precisione più alta, la coda a precisione più bassa.

Modello di riferimento pubblicato su HuggingFace:
**[vagrillo/Qwen35B-A4B-MOExpMixedQuant](https://huggingface.co/vagrillo/Qwen35B-A4B-MOExpMixedQuant)**
(Qwen3.6-35B-A3B, 4 pool: top-16 Q8_0 / 17-64 Q6_K / 65-192 Q4_K / 193-256 Q2_K
per layer, shared expert a Q8_0 — ~21 GB contro 37 GB della Q8_0 uniforme).

## Composizione

Due componenti che lavorano insieme:

### 1. Quantizzazione con split (`llama-quantize --split-experts`)

```
llama-quantize --allow-requantize \
  --imatrix imatrix_unsloth.gguf \
  --split-experts expert-order.txt \
  Qwen3.6-35B-A3B-Q8_0.gguf Qwen3.6-35B-A3B-mixed-expert.gguf Q8_0
```

`expert-order.txt`: una riga per layer MoE, 256 id di esperti separati da
virgola in ordine di rank d'uso (decrescente). Per ogni tensore di esperti
routati del sorgente (`ffn_gate_up_exps`, oppure `ffn_gate_exps` + `ffn_up_exps`
se il sorgente usa gate/up separati, e `ffn_down_exps`) produce **4 tensori
pool** (`_hi`/`_mid`/`_lo`/`_tail`, 16/48/128/64 esperti) quantizzati con il
tipo della fascia, più **8 tabelle F32 [n_expert]** per layer
(`ffn_exps_local_0..3`, `ffn_exps_mask_0..3`): `local_p[e]` = indice locale
dell'esperto `e` nel pool `p` (0 se fuori pool), `mask_p[e]` = 1.0 se nel pool.
L'imatrix, essendo per-esperto (`[n_cols, n_expert]`), viene affettata lungo
l'asse esperti: ogni pool è quantizzato con la sua porzione di imatrix.
Tutti gli altri tensori (attention, embedding, output, shared expert, router,
norme) vengono copiati/quantizzati dal percorso standard.

### 2. Grafico con pool (`build_moe_ffn`, pool mode)

Il loader di `qwen35moe` rileva i tensori pool; `build_moe_ffn` riceve i 4 pool
e le tabelle e, a routing invariato (stesso softmax, stesso top-20, stessa
espansione e stesso threshold), per ogni pool:

- rimappa gli id del router all'indice locale: `ids_p = cast_i32(get_rows(local_p, ids))`
- maschera i pesi: `w_p = weights * get_rows(mask_p, ids)` (gli esperti fuori
  pool valutano l'esperto locale 0 con peso 0: contributo nullo, shape statiche)
- calcola la FFN del pool (`mul_mat_id` gate_up o gate/up separati → swiglu → down)
- somma i parziali di tutti i pool

Il risultato matematico è identico al compute non pooled; cambia solo la
precisione di ciascun esperto. Il tracking dell'uso esperti
(`LLAMA_MOE_EXPERT_USAGE_FILE`, vedi sotto) continua a funzionare perché il
routing non è toccato.

Nota sulle prestazioni: il compute per-pool valuta `n_used` righe per pool
(4 × 20 valutazioni di esperto per token invece di 20) — l'overhead sulle FFN
è ~4× in decode. Le versioni successive possono ridurlo con budget per pool
basati sul p99 dei conteggi per token (argsort per pool + viste a budget fisso).

## Tracciamento dell'uso esperti (per generare il ranking)

`LLAMA_MOE_EXPERT_USAGE_FILE=path.json` attiva il conteggio cumulativo
per (layer, esperto) di tutti gli esperti mantenuti (peso post-espansione > 1e-6):

```bash
LLAMA_MOE_EXPERT_USAGE_FILE=usage.json LLAMA_MOE_EXPERT_USAGE_EVERY=256 \
  llama-server -m model.gguf --moe-experts 20 --moe-expert-threshold 0.95 ...
```

Il JSON viene riscritto in modo atomico ogni `EVERY` token:

```json
{"format": "moe-expert-usage-v2", "total_tokens": 441745,
 "layer_tokens": {"0": 441745, ...},
 "layers": {"0": {"n_expert": 256, "counts": [1203, 87, ...]}, ...}}
```

`counts[E]` = quante volte l'esperto `E` è stato mantenuto dal router su tutti
i token contati. Il readback avviene nella **eval callback dello sched**
(sync per-tensor, pattern `moe_chunk_cb`): il readback a fine grafo perde ubatch
di decode per aliasing dei buffer (CUDA graphs + topologie concorrenti).
Un poller esterno accumula i delta in un totale persistente tra i run.
Logging: `LLAMA_LOG_INFO` è filtrato di default in questo fork — gli eventi di
dump non compaiono nei log standard.

## Workload di riferimento e risultati

Ranking raccolto su GPQA-Diamond (93 domande, replay greedy, reasoning budget
8192, concorrenza 4) su V100-32GB: 441.745 token, 20.591.738 selezioni, tutti i
40 layer validati a ~11,5 selezioni/token. Metodo, caveats (bias di dominio,
distribuzione quasi piatta in coda, shared expert escluso) e file:
repository [vagrillo/AgrillaMoE](https://github.com/vagrillo/AgrillaMoE),
directory `expert-rank/` (`EXPERT-RANK.md`).
