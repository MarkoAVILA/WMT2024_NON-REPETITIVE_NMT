#!/bin/bash

# === CONFIGURACIÓN GENERAL ===
BASE_DIR="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE"
TRAIN_DIR="$BASE_DIR/data/train/corpus/train_jiji"
FAST_ALIGN_DIR="fast_align/build"
ALIGN_DIR="$BASE_DIR/align"
SCRIPT_DIR="$BASE_DIR/data/train/WMT2024_NON-REPETITIVE_official_data/scripts"
ALIGN_FILE="$ALIGN_DIR/data"

# === FUNCIÓN: Entrenamiento de alineación con fast_align ===
align_train() {
    echo "=== Generando alineación con fast_align ==="
    mkdir -p "$ALIGN_DIR"

    # Paso 1: Tokenizar y concatenar
    python "$SCRIPT_DIR/tokenize_and_paste.py" \
        "$TRAIN_DIR/ja.syn.txt" \
        "$TRAIN_DIR/en.syn.txt" > "$ALIGN_FILE"

    # Paso 2: Alineación directa y reversa
    $FAST_ALIGN_DIR/fast_align -i "$ALIGN_FILE" -d -o -v > "$ALIGN_DIR/forward" 2> "$ALIGN_DIR/forward.log" &
    $FAST_ALIGN_DIR/fast_align -i "$ALIGN_FILE" -d -o -v -r > "$ALIGN_DIR/reverse" 2> "$ALIGN_DIR/reverse.log" &
    
    wait

    # Paso 3: Combinación con atools
    $FAST_ALIGN_DIR/atools -i "$ALIGN_DIR/forward" -j "$ALIGN_DIR/reverse" -c grow-diag-final-and > "$ALIGN_DIR/data.gdfa"
    echo "=== Alineación terminada ==="
}

# === FUNCIÓN: Penalizaciones con penaltiesV2.py ===
run_penalty() {
    local src_file=$1
    local tgt_file=$2
    local ali_file=$3
    local ref_file=$4
    local out_file=$5
    local log_file=$6

    echo "=== Ejecutando penaltiesV2.py en $src_file ==="
    nohup python penaltiesV2.py \
        --src "$src_file" \
        --tgt "$tgt_file" \
        --ali "$ali_file" \
        --ref "$ref_file" \
        --max 0 --min 3 \
        > "$out_file" 2> "$log_file" &
}

# === EJECUCIÓN ===
align_train

# Penalización para traducción con sinónimos (syn)
run_penalty \
    "$TRAIN_DIR/ja.syn.txt" \
    "$TRAIN_DIR/en.syn.txt" \
    "$ALIGN_DIR/data.gdfa" \
    "$TRAIN_DIR/en.txt" \
    "$TRAIN_DIR/penalty2" \
    "$TRAIN_DIR/log"

# Penalización para referencia original (comentado como ejemplo)
# run_penalty \
#     "$dir_raw/train.en" \
#     "$dir_raw/train.fr" \
#     "$dir_raw/train.fr.gdfa" \
#     "$dir_raw_syn/train.fr" \
#     "$dir_raw/penalty" \
#     "$dir_raw/log"

