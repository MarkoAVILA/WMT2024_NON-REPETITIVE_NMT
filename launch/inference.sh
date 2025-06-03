#!/bin/bash

echo "=== Fine-tuning and inference with Systran Models ==="

# === FUNCIÓN: Inferencia con transformer-infer ===
run_infer() {
    local ckpt_path=$1
    local src_vocab=$2
    local tgt_vocab=$3
    local input_file=$4
    local output_file=$5
    local log_file=$6

    echo "--- Running inference: $input_file --> $output_file"
    nohup transformer-infer \
        --ckpt "$ckpt_path" \
        --device "cuda" \
        --src_vocab "$src_vocab" \
        --tgt_vocab "$tgt_vocab" \
        --input "$input_file" \
        --output "$output_file" \
        decode &> "$log_file" &
}

# === Configuración de modelos ===

# 🔹 Modelo 1 (Systran original)
MODEL1_DIR="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/models/model_base/systran_models/model_enja"
DIR_DATA="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/data/train/corpus/train_jiji"
CKPT1="$MODEL1_DIR/checkpoint-137229.pt"
VOCAB1="$MODEL1_DIR/joint_vocab_enja20M-50000.en_ja.v7.converted"

# Inferencia: EN ➡ JA.SYN
run_infer "$CKPT1" "$VOCAB1" "$VOCAB1" "$DIR_DATA/en.tok" "$DIR_DATA/ja.syn.tok" "logs/infer_en_to_ja_syn_model1.log"

# 🔹 Modelo 2 (Systran convertido)
MODEL2_DIR="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/models/model_base/systran_models/model_converted"
PRED="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/predictions"
CKPT2="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/models/model_base/checkpoint-126517.pt"
VOCAB2="$MODEL2_DIR/joint_vocab_enja30M-56000.ja_en.v10.converted"

# Inferencia: JA ➡ EN (base)
run_infer "$CKPT2" "$VOCAB2" "$VOCAB2" "$DIR_DATA/ja.tok" "$PRED/pred_base.tok" "logs/infer_ja_to_en_base_model2.log"

# Inferencia: JA.SYN ➡ EN.SYN
run_infer "$CKPT2" "$VOCAB2" "$VOCAB2" "$DIR_DATA/ja.syn.tok" "$PRED/en.syn.tok" "logs/infer_ja_syn_to_en_syn_model2.log"
