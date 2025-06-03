import contextlib

import torch
import sacrebleu
from sacrebleu import corpus_bleu
# import sys
# sys.path.append("/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/pytorch-transformer/src/")
# import transformer
from transformer.beam_search import beam_search
from transformer.data import BOS_TOKEN, EOS_TOKEN, PAD_ID
from transformer.tensor_parallel.cross_entropy import VocabParallelCrossEntropyLoss
from transformer.preprocessing import *
from transformer.c_reps import counting

def open_file(file):
    with open(file, 'r') as f:
        l = [i.strip() for i in f]
    return l

def evaluate(
    model,
    bpe_tgt,
    config,
    target_vocabulary_path,
    dataset,
    target_vocabulary,
    target_vocabulary_rev,
    device,
    enable_mixed_precision=False,
    predictions_path=None,
    output_score=False,
    tensor_parallel=False,
    vocab_tensor_parallel=False,
):
    if tensor_parallel and vocab_tensor_parallel:
        ce_loss = VocabParallelCrossEntropyLoss(
            ignore_index=PAD_ID,
            reduction="none",
        )
    else:
        ce_loss = torch.nn.CrossEntropyLoss(
            ignore_index=PAD_ID,
            reduction="none",
        )
    VOCAB_TOK=target_vocabulary_path #"/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/model_converted/joint_vocab_enja30M-56000.ja_en.v10.converted"
    MODEL_TOK=bpe_tgt #"/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/model_converted/bpe_enja30M-28000.en"
    CONFIG=config #"/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/Models/model_generic/config.json"
    src_ja = open_file("/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/corpus_tagged/test/ja.txt")
    tok_ja = open_file("/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/tokens_true")
    P = Preprocessing(n_symbols=32000, vocab_path=VOCAB_TOK, model_path=MODEL_TOK, config=CONFIG, from_systran=True)
    bos, eos = target_vocabulary[BOS_TOKEN], target_vocabulary[EOS_TOKEN]

    total_ce_loss = 0
    num_tokens = 0
    ce_loss_per_example = []
    l_target = []
    pred_=[]
    with torch.autocast(
        device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enable_mixed_precision,
    ), open(
        predictions_path, "w"
    ) if predictions_path else contextlib.nullcontext() as f:
        for batch in dataset:
            source = batch["source"]
            target_in = batch["target_in"]
            target_out = batch["target_out"]
            for t in target_out:
                    l = []
                    for token_id in t:
                        if target_vocabulary_rev[token_id]!="</s>":
                            l.append(target_vocabulary_rev[token_id])
                        else:
                            break
                    l_target.append(" ".join(l))
            with torch.no_grad():
                logits = model(source, target_in)
                if tensor_parallel and vocab_tensor_parallel:
                    # Tensor Parallel
                    loss = ce_loss(logits.float(), target_out)
                else:
                    loss = ce_loss(logits.transpose(1, 2), target_out)
                if output_score:
                    ce_loss_per_example.extend(loss.sum(dim=1).tolist())
                total_ce_loss += loss.sum().item()
                num_tokens += target_out.ne(PAD_ID).sum().item()

            if f is not None:
                result = beam_search(
                    model, source, bos, eos, parallel_output=(not vocab_tensor_parallel)
                )
                for hypotheses in result:
                    tokens = hypotheses[0][1]
                    if tokens and tokens[-1] == eos:
                        tokens.pop(-1)
                    tokens = [target_vocabulary_rev[token_id] for token_id in tokens]
                    pred_.append(" ".join(tokens))
                    f.write(" ".join(tokens) + "\n")
    res = total_ce_loss / num_tokens if num_tokens else 0.0
    if output_score:
        res = (res, ce_loss_per_example)
    pred_txt = P.detokenization(pred_)
    target_txt = P.detokenization(l_target)
    reps_sentences, reps_tokens = counting(src_ja, pred_txt, tok_ja)
    BLEU_ = corpus_bleu(pred_txt, [target_txt]).score
    return res, BLEU_, reps_sentences, reps_tokens
