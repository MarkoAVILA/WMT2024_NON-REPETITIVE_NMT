#!/bin/bash

echo "Fine-Tuning Systran Models - Experimentos varios"

# ─────────────────────────────────────────────────────────────
# Configuración común para todos los experimentos
VOCAB_TOK="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/model_converted/joint_vocab_enja30M-56000.ja_en.v10.converted"
MODEL_TOK_2="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/model_converted/bpe_enja30M-28000.en"
CONFIG="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/model_converted/config.json"

# Directorio de validación
VALID_DIR="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024"

# ─────────────────────────────────────────────────────────────
# Entrenamiento 1: Modelo básico con corpus_tagged
DIR_DATA="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/corpus_tagged/"
nohup transformer-train \
  --src $DIR_DATA/train/ja.tok \
  --tgt $DIR_DATA/train/en.tok \
  --src_valid $DIR_DATA/test/ja.tok \
  --tgt_valid $DIR_DATA/test/en.tok \
  --src_vocab $VOCAB_TOK --tgt_vocab $VOCAB_TOK \
  --batch_size 200 --bpe_tgt $MODEL_TOK_2 \
  --config $CONFIG \
  --save_dir Models_final/model_tag/checkpoints/ \
  > logs/log_tag &

# Entrenamiento 2: Variante con transformer-train2
nohup transformer-train2 \
  --src $DIR_DATA/train/ja.tok \
  --tgt $DIR_DATA/train/en.tok \
  --src_valid $DIR_DATA/test/ja.tok \
  --tgt_valid $DIR_DATA/test/en.tok \
  --src_vocab $VOCAB_TOK --tgt_vocab $VOCAB_TOK \
  --batch_size 200 --bpe_tgt $MODEL_TOK_2 \
  --config $CONFIG \
  --save_dir Models_final/model_tag2/checkpoints/ \
  > logs/log_tag2 &

# ─────────────────────────────────────────────────────────────
# Entrenamiento 3: Penalización con POS tags (pen2.pos)
DIR_DATA="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/corpus/train_jiji"
nohup transformer-trainpen \
  --src $DIR_DATA/ja.syn.tok \
  --tgt $DIR_DATA/en.syn.tok \
  --src_valid $VALID_DIR/ja.tok \
  --tgt_valid $VALID_DIR/en.tok \
  --src_vocab $VOCAB_TOK --tgt_vocab $VOCAB_TOK \
  --bpe_tgt $MODEL_TOK_2 \
  --pos $DIR_DATA/pen2.pos --alpha 0.999999 \
  --batch_size 200 --config $CONFIG \
  --save_dir Models/model_pen2/checkpoints/ \
  > logs/log_train_pen2 &

# ─────────────────────────────────────────────────────────────
# Entrenamiento 4: Variante reward alpha=1 con corpus d1
DIR_DATA="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/corpus/d1"
nohup transformer-trainpen2 \
  --src $DIR_DATA/ja.syn.tok \
  --tgt $DIR_DATA/en.syn.tok \
  --src_valid $VALID_DIR/ja.tok \
  --tgt_valid $VALID_DIR/en.tok \
  --src_vocab $VOCAB_TOK --tgt_vocab $VOCAB_TOK \
  --bpe_tgt $MODEL_TOK_2 \
  --pos $DIR_DATA/pen2.pos --alpha 1 \
  --batch_size 200 --config $CONFIG \
  --save_dir Models_final/model_reward_final/checkpoints/ \
  > logs/log_train_reward &

# ─────────────────────────────────────────────────────────────
# Entrenamiento 5: Penalización directa (modelo final d1)
nohup transformer-trainpen \
  --src $DIR_DATA/ja.tok \
  --tgt $DIR_DATA/en.tok \
  --src_valid $VALID_DIR/ja.tok \
  --tgt_valid $VALID_DIR/en.tok \
  --src_vocab $VOCAB_TOK --tgt_vocab $VOCAB_TOK \
  --bpe_tgt $MODEL_TOK_2 \
  --pos $DIR_DATA/pos --alpha 0.9999999 \
  --batch_size 200 --config $CONFIG \
  --save_dir Models_final/model_d1/checkpoints/ \
  > logs/log_d1_final &

# ─────────────────────────────────────────────────────────────
# P1 - Placeholder para experimento adicional
# Puedes completar aquí con otro bloque si lo deseas

echo "All fine-tuning jobs have been launched!"
