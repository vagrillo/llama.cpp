#!/bin/bash
# GPQA-diamond paired benchmark su vast.ai con il branch moe-expansion
# Config: Qwen3.6-35B-A3B (UD-Q6_K_XL) | N=20, T=0.8, layer 29..39, decay 0.99..0.50
#
# Uso (sull'istanza vast.ai, dentro tmux):
#   bash vastai-gpqa-bench.sh setup    # build + modello (una volta, ~30 min)
#   bash vastai-gpqa-bench.sh server-exp    # server con espansione (terminale 1)
#   bash vastai-gpqa-bench.sh server-native # server nativo K=8 (terminale 2)
#   bash vastai-gpqa-bench.sh eval exp      # evalscope GPQA-diamond vs server attivo
#   bash vastai-gpqa-bench.sh eval native
set -e

MODEL_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
MODEL_FILE="Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"
MODEL_DIR="$HOME/models"
LLAMA_DIR="/workspace/llama.cpp"
PORT=9080
CTX=42768          # alza a 65536 se hai >= 64 GB di VRAM
MAX_TOKENS=32768   # protocollo ds4: max_tokens >= 8192; i troncati si rilanciano

EXP_FLAGS=(--moe-experts 20 --moe-expert-threshold 0.8
           --moe-expert-layer-start 29 --moe-expert-layer-end 40
           --moe-expert-decay-end 0.5)

stage_setup() {
    apt update && apt install -y build-essential cmake git libcurl4-openssl-dev python3-pip
    [ -d "$LLAMA_DIR" ] || git clone -b moe-expansion https://github.com/vagrillo/llama.cpp "$LLAMA_DIR"
    cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA_DIR/build" -j"$(nproc)" --target llama-server llama-cli

    pip install -U huggingface_hub
    mkdir -p "$MODEL_DIR"
    hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"

    pip install -U evalscope
    echo "== setup completato"
}

stage_server() {
    local name="$1"; shift
    mkdir -p "$HOME/logs"
    local flags=()
    [ "$name" = "exp" ] && flags=("${EXP_FLAGS[@]}")
    "$LLAMA_DIR/build/bin/llama-server" \
        -m "$MODEL_DIR/$MODEL_FILE" \
        -ngl 999 -c "$CTX" --temp 0 --jinja \
        --host 0.0.0.0 --port "$PORT" \
        "${flags[@]}" 2>&1 | tee "$HOME/logs/server-$name.log"
}

stage_eval() {
    local name="$1"
    evalscope eval \
        --model "Qwen3.6-35B-A3B-$name" \
        --eval-type openai_api \
        --api-url "http://127.0.0.1:$PORT/v1" \
        --datasets gpqa_diamond \
        --generation-config '{"temperature":0,"max_tokens":32768, "timeout":3600, "stream":true}' \
         --eval-batch-size 1 \
        --work-dir "$HOME/gpqa-$name" \
        --use-cache "$HOME/gpqa-$name" --rerun-review
}

case "${1:-}" in
    setup)         stage_setup ;;
    server-exp)    stage_server exp ;;
    server-native) stage_server native ;;
    eval)          stage_eval "$2" ;;
    *) echo "uso: $0 {setup|server-exp|server-native|eval exp|eval native}"; exit 1 ;;
esac

