#!/bin/bash
# GPQA-diamond paired benchmark su DeepSeek-V4-Flash-0731 (unsloth GGUF, Q2)
# con il branch moe-expansion - pensato per macchine con 2x GPU 48GB (96GB total).
#
# Uso principale (dentro tmux):
#   bash benchds4.sh setup       # build + modello + evalscope (una volta)
#   bash benchds4.sh pipeline    # <-- lancia NATIVO prima, poi ESPANSO, di seguito
#
# Gli host potrebbero chiamarla anche via:
#   bash benchds4.sh server-exp | server-native | eval exp | eval native
#   bash benchds4.sh eval-divergent  # rifà SOLO le domande in cui il nativo ha battuto l'espanso (server EXPANSO attivo)
#
# I parametri di espansione si scelgono al lancio via environment, es.:
#   EXPERTS=12 THRESHOLD=0.8 bash benchds4.sh pipeline
#   EXPERTS=16 THRESHOLD=0.7 DECAY=0.3 LAYER_START=0.5 bash benchds4.sh pipeline
#
# Se il Q2 non entra in 96GB di VRAM: OFFLOAD_EXPERTS_CPU=1 mette i tensori
# degli esperti su RAM (kernel MoE su CPU, resto su GPU) - piu lento ma funziona.
set -e

# --- modello: unsloth = quant garantite compatibili con llama.cpp -----------
# UD-IQ2_M: 90.9 GB, split in 3 shard (qualita' 2-bit migliore di IQ2_XXS).
# Su 2x48GB (96GB VRAM) resta pochissimo spazio per KV/buffer: di default gli
# esperti (grosso modo 80GB) stanno su RAM via -ot e serve system RAM >= 100GB.
# Per provare tutto su GPU: OFFLOAD_EXPERTS_CPU=0 e contesto ridotto (es. CTX=8192).
# Alternative: UD-IQ2_XXS 90.9GB, UD-IQ1_M 86.9GB (qualita' inferiore).
MODEL_REPO="unsloth/DeepSeek-V4-Flash-0731-GGUF"
MODEL_DIR_SUB="UD-IQ2_M"
MODEL_FILE="$MODEL_DIR_SUB/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf"

MODEL_DIR="$HOME/models"
LLAMA_DIR="/workspace/llama.cpp"
PORT=9080
CTX=42768                # contesto totale (con --parallel 1 resta tutto a 1 richiesta)
MAX_TOKENS=32768
NGL=999                  # layer su GPU (999 = tutti)
PARALLEL=1               # 1 slot = tutto il contesto per la singola richiesta
OFFLOAD_EXPERTS_CPU=1    # default 1: il Q3_K_M (128GB) supera i 96GB di VRAM
                         # -> tensori esperti su RAM via -ot (richiede >=130GB system RAM)

# --- parametri di espansione (override da environment) ----------------------
# DS4-Flash: 6 esperti nativi (top-6). N e' il budget massimo; con T<=1 non
# scende mai sotto il nativo; i primi layer hash-routed restano nativi in ogni caso.
EXPERTS="${EXPERTS:-12}"
THRESHOLD="${THRESHOLD:-0.8}"
DECAY="${DECAY:-0.5}"
LAYER_START="${LAYER_START:-}"   # es. 0.5 (seconda meta dei layer) oppure indice
LAYER_END="${LAYER_END:-}"
NO_DECAY="${NO_DECAY:-0}"
# renorm: auto = segue la normalizzazione stock del modello (DeepSeek-V4 senza
# expert_weights_norm=true -> nessuna renormalizzazione, scala grezza dei punteggi)
RENORM="${RENORM:-auto}"

EXP_FLAGS=(--moe-experts "$EXPERTS" --moe-expert-renorm "$RENORM")
[ "$THRESHOLD" != "0" ] && EXP_FLAGS+=(--moe-expert-threshold "$THRESHOLD")
[ "$DECAY" != "none" ] && EXP_FLAGS+=(--moe-expert-decay-end "$DECAY")
[ "$NO_DECAY" = "1" ] && EXP_FLAGS+=(--moe-no-expert-decay)
[ -n "$LAYER_START" ] && EXP_FLAGS+=(--moe-expert-layer-start "$LAYER_START")
[ -n "$LAYER_END" ] && EXP_FLAGS+=(--moe-expert-layer-end "$LAYER_END")

SERVER_PID=""

wait_health() {
    echo "== attesa readiness del server (load di ~90GB, puo' richiedere minuti)..."
    for i in $(seq 1 360); do
        if curl -s -m 5 "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then
            echo "== server pronto"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: server non ready entro il timeout" >&2
    return 1
}

start_server() {
    local name="$1"; shift
    local flags=()
    [ "$name" = "exp" ] && flags=("${EXP_FLAGS[@]}")
    local ot=()
    [ "$OFFLOAD_EXPERTS_CPU" = "1" ] && ot=(-ot ".ffn_.*_exps.=CPU")
    mkdir -p "$HOME/logs"
    echo "== avvio server [$name] (log: ~/logs/server-ds4-$name.log)"
    nohup "$LLAMA_DIR/build/bin/llama-server" \
        -m "$MODEL_DIR/$MODEL_FILE" \
        -ngl "$NGL" -c "$CTX" --temp 0 --jinja --parallel "$PARALLEL" \
        --split-mode layer \
        --host 0.0.0.0 --port "$PORT" \
        "${ot[@]}" "${flags[@]}" > "$HOME/logs/server-ds4-$name.log" 2>&1 &
    SERVER_PID=$!
}

stop_server() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
    sleep 3
}

stage_setup() {
    apt update && apt install -y build-essential cmake git libcurl4-openssl-dev python3-pip
    [ -d "$LLAMA_DIR" ] || git clone -b moe-expansion https://github.com/vagrillo/llama.cpp "$LLAMA_DIR"
    cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA_DIR/build" -j"$(nproc)" --target llama-server llama-cli

    pip install -U huggingface_hub
    mkdir -p "$MODEL_DIR"
    # scarica solo la cartella della quant scelta (tutti gli shard dello split)
    hf download "$MODEL_REPO" --include "$MODEL_DIR_SUB/*" --local-dir "$MODEL_DIR"
    [ -f "$MODEL_DIR/$MODEL_FILE" ] || { echo "ERROR: manca $MODEL_FILE"; exit 1; }
    echo "== GGUF: $MODEL_FILE (UD-IQ2_M, ~90.9GB, 3 shard)"

    pip install -U evalscope
    echo "== setup completato"
}

stage_server() {
    local name="$1"
    start_server "$name"
    wait_health
    echo "== server [$name] in ascolto su :$PORT - Ctrl+C per fermare"
    wait "$SERVER_PID"
}

stage_eval() {
    local name="$1"
    evalscope eval \
        --model "DeepSeek-V4-Flash-0731-$name" \
        --eval-type openai_api \
        --api-url "http://127.0.0.1:$PORT/v1" \
        --datasets gpqa_diamond \
        --generation-config '{"temperature":0,"max_tokens":'"$MAX_TOKENS"',"timeout":3600,"stream":true}' \
        --eval-batch-size 1 \
        --work-dir "$HOME/gpqa-ds4-$name" \
        --use-cache "$HOME/gpqa-ds4-$name" --rerun-review
}

# rifà SOLO le domande divergenti scelte con SELECT (default: quelle dove il
# nativo ha battuto l'espanso), contro il SERVER ESPANSO attivo, con budget pieno.
# richiede eval_divergent.py nella stessa directory dello script.
stage_eval_divergent() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # fallback: se i work-dir ds4 non esistono usa i nomi del bench.sh originale
    EXP_DIR="${EXP_DIR:-$HOME/gpqa-ds4-exp}"
    NAT_DIR="${NAT_DIR:-$HOME/gpqa-ds4-native}"
    [ -d "$EXP_DIR/predictions" ]  || EXP_DIR="$HOME/gpqa-exp"
    [ -d "$NAT_DIR/predictions" ]  || NAT_DIR="$HOME/gpqa-native"
    echo "work-dir: exp=$EXP_DIR  nat=$NAT_DIR"
    python3 "$script_dir/eval_divergent.py" \
        --exp-dir "$EXP_DIR" \
        --nat-dir "$NAT_DIR" \
        --port "$PORT" \
        --newcap "$MAX_TOKENS" \
        --select "${SELECT:-nat_better}" \
        --out "$HOME/gpqa-ds4-divergent-redo.json"
}

# pipeline FULL-GPU: niente esperti su CPU. il modello (90.9GB) occupa quasi
# tutto il VRAM dei 96GB: serve un contesto piccolo, altrimenti OOM al load.
# tradeoff: veloce (tpot ~12ms) ma il contesto corto puo' troncare le risposte
# lunghe (rilancia con CTX_GPU/ MAX_TOKENS_GPU per regolare).
stage_pipeline_gpu() {
    OFFLOAD_EXPERTS_CPU=0
    CTX="${CTX_GPU:-16384}"
    MAX_TOKENS="${MAX_TOKENS_GPU:-12288}"
    echo "== pipeline FULL-GPU: esperti su GPU, CTX=$CTX, max_tokens=$MAX_TOKENS"
    echo "   (se OOM al load: CTX_GPU=8192 MAX_TOKENS_GPU=8192, oppure usa la pipeline con offload CPU)"
    stage_pipeline
}

# pipeline: NATIVO prima, poi ESPANSO, tutto in un comando
stage_pipeline() {
    echo "########## 1/2 NATIVO (top-6) ##########"
    start_server native
    wait_health
    stage_eval native
    stop_server

    echo "########## 2/2 ESPANSO (N=$EXPERTS, T=$THRESHOLD) ##########"
    start_server exp
    wait_health
    stage_eval exp
    stop_server

    echo "=============================================================="
    echo "pipeline completata. risultati in:"
    echo "  ~/gpqa-ds4-native   e   ~/gpqa-ds4-exp"
    echo "confronto:  python3 compare_gpqa.py ~/gpqa-ds4-exp ~/gpqa-ds4-native"
}

case "${1:-}" in
    setup)         stage_setup ;;
    server-exp)    start_server exp;    wait_health; echo "server pronto :$PORT - Ctrl+C per fermare"; wait "$SERVER_PID" ;;
    server-native) start_server native; wait_health; echo "server pronto :$PORT - Ctrl+C per fermare"; wait "$SERVER_PID" ;;
    eval)          stage_eval "$2" ;;
    eval-divergent) stage_eval_divergent ;;
    pipeline)      stage_pipeline ;;
    pipeline-gpu)  stage_pipeline_gpu ;;
    *) echo "uso: $0 {setup|pipeline|pipeline-gpu|server-exp|server-native|eval exp|eval native|eval-divergent}"; exit 1 ;;
esac
