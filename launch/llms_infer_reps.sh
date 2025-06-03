#!/bin/bash

echo "Using LLMs for Repetitions Fluency in NMT..."
echo "We are going to prompt ChatGPT and run NLLB for the repetitions fluency task :)"

DATA_DIR="/nfs/RESEARCH/avila/Challenges/WMT2024_NON-REPETITIVE/JIJI_CORPUS_2024"
INPUT_FILE="$DATA_DIR/wmt2024.test.raw.ja"

# Define models and settings
declare -A MODELS

MODELS[gpt3.5]="python3 /nfs/RESEARCH/avila/Projects/TEXT2TEXT/Reducing_Reps_NMT/NMT_REP/finetuning/infer_gpt3.5.py generate \
  --p \"Translate the following text from Japanese to English, ensuring that the translated output maintains coherence and fluency while minimizing the repetition of words or phrases. Pay attention to using synonyms, varied sentence structures, and appropriate linguistic devices to enhance the overall quality of the translation. Feel free to creatively adapt the language to achieve a natural and engaging tone in the target language. I want you to only reply the traduction, do not write explanations\" \
  --input_data $INPUT_FILE --output_dir $DATA_DIR/pred-gpt.txt"

MODELS[nllb]="python3 /nfs/RESEARCH/avila/Projects/TEXT2TEXT/Reducing_Reps_NMT/NMT_REP/finetuning/infer_nllb.py run \
  --input_file $INPUT_FILE --output_dir $DATA_DIR/nllb-pred.txt \
  --model_name /nfs/RESEARCH/avila/Projects/TEXT2TEXT/Reducing_Reps_NMT/NMT_REP/nllb-200-distilled-600M-ct2 \
  --langues \"Japanese-English\" --device cuda"

# Run each model inference in background 
for model in "${!MODELS[@]}"; do
  echo "Running inference for $model ..."
  nohup ${MODELS[$model]} > "$DATA_DIR/log_${model}_infer" 2>&1 &
done
