#!/bin/bash
echo "We are going to convert a PN9 model Systran to pytorch :)"
SYSTRAN_MODELS="~avila/models/model_base/systran_models"
python3 pytorch-transformer/src/transformer/pn9_tf/convert.py --model_path $SYSTRAN_MODELS/model_ja_en_systran/SSJYA_enja_SingleNumbatNFA_214_bd1ace317c-c524e --output_model_path $SYSTRAN_MODELS/model_converted/ --convert_vocab
echo "Model converted!"
mkdir ~avila/models/model_base
echo "Copying checkpoint pytorch in model_base"
cp -r $SYSTRAN_MODELS/model_converted/*.pt ~avila/models/model_base/
echo "done"
