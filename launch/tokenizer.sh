#!/bin/bash

# === CONFIGURACIÓN ===
BPE_JA="model_converted/bpe_enja30M-28000.ja"
BPE_EN="model_converted/bpe_enja30M-28000.en"
VOCAB_TOK="model_converted/joint_vocab_enja30M-56000.ja_en.v10.converted"
CONFIG="model_converted/config.json"
DATA="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/data"
TRN_DATA="$DATA/train/corpus/train_jiji"
TST_DATA="$DATA/test/JIJI_CORPUS_2024"
TAGGED="$DATA/corpus_tagged/test"
DIR="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/predictions"

# === TOKENIZACIÓN ===
declare -A TOKENIZATION_TASKS=(
  ["$TRN_DATA/ja.txt"]="$TRN_DATA/ja.tok:source:$BPE_JA"
  ["$TRN_DATA/ja.syn.txt"]="$TRN_DATA/ja.syn.tok:source:$BPE_JA"
  ["$TRN_DATA/en.txt"]="$TRN_DATA/en.tok:target:$BPE_EN"
  ["$TST_DATA/wmt2024.test.raw.ja"]="$TST_DATA/ja.tok:source:$BPE_JA"
  ["$TST_DATA/wmt2024.test.raw.en"]="$TST_DATA/en.tok:source:$BPE_EN"
  ["$TAGGED/ja.txt"]="$TAGGED/ja.tok:source:$BPE_JA"
  ["$TAGGED/en.txt"]="$TAGGED/en.tok:source:$BPE_EN"
)

echo "=== TOKENIZING ==="
for input in "${!TOKENIZATION_TASKS[@]}"; do
  IFS=":" read -r output ty model <<< "${TOKENIZATION_TASKS[$input]}"
  echo "Tokenizing $input -> $output"
  python3 preprocessing.py tokenization_bpe_path \
    --vocab_path "$VOCAB_TOK" \
    --model_path "$model" \
    --config "$CONFIG" \
    --ty "$ty" \
    --input_file "$input" \
    --output_file "$output"
done


# === DETOKENIZACIÓN ===
echo "=== DETOKENIZING ==="
declare -A DETOKENIZATION_TASKS=(
  ["$DIR/pred_base.tok"]="$DIR/pred_base.txt"
  ["$DIR/predictions-128000.out"]="$DIR/predictions-128000.txt"
  ["$DIR/predictions-138500.out"]="$DIR/predictions-138500.txt"
  ["/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/models/models_finetuned/model_tag2/checkpoints/predictions-136000.out"]="$DIR/predictions-136000.txt"
  ["/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/models/models_finetuned/model_ft/checkpoints/predictions-133000.out"]="$DIR/predictions-133000_ft.txt"
  ["/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/models/models_finetuned/p1/checkpoints/predictions-127000.out"]="$DIR/predictions-127000.txt"
)

for input in "${!DETOKENIZATION_TASKS[@]}"; do
  output="${DETOKENIZATION_TASKS[$input]}"
  echo "Detokenizing $input -> $output"
  python3 preprocessing.py detokenization_from_file_bpe_path \
    --vocab_path "$VOCAB_TOK" \
    --model_path "$BPE_EN" \
    --config "$CONFIG" \
    --ty "target" \
    --file_input "$input" \
    --file_output "$output"
done
