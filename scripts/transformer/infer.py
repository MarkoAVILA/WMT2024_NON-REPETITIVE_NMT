import argparse
import sys

import torch
# import sys
# sys.path.append("/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/pytorch-transformer/src/")
# import transformer
from transformer.beam_search import beam_search
from transformer.config import default_config, load_config, merge_config
from transformer.data import BOS_TOKEN, EOS_TOKEN, load_vocabulary
from transformer.dataset import create_inference_dataset
from transformer.evaluate import evaluate
from transformer.model import Transformer
from transformer.tensor_parallel.initialize import (
    set_model_parallel_rank,
    set_model_parallel_world_size,
)
from transformer.utils import fh_out, init_logger


def main():
    init_logger()

    default_config_infer = default_config.get("infer")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file for inference options")

    data_options = parser.add_argument_group("Data options")
    parser.add_argument("--input", help="Path to the input (source) file")
    parser.add_argument(
        "--ref", help="Path to the reference (target) file (used for scoring)"
    )
    parser.add_argument("--output", help="Path to the output file")
    data_options.add_argument(
        "--src_vocab", required=True, help="Path to the source vocabulary"
    )
    data_options.add_argument(
        "--tgt_vocab", required=True, help="Path to the target vocabulary"
    )
    data_options.add_argument(
        "--batch_size",
        type=int,
        default=default_config_infer.get("batch_size"),
        help="Batch size",
    )

    model_options = parser.add_argument_group("Model options")
    model_options.add_argument("--ckpt", required=True, help="Path to the checkpoint")
    model_options.add_argument("--device", default="cpu", help="Device to use")

    subparsers = parser.add_subparsers(
        help="Inference type command (decode or score)", dest="cmd", required=True
    )

    parser_decode = subparsers.add_parser("decode", help="Decode from source file")
    decoding_options = parser_decode.add_argument_group("Decoding options")
    decoding_options.add_argument(
        "--beam_size",
        type=int,
        default=default_config_infer.get("beam_size"),
        help="Beam size",
    )
    decoding_options.add_argument(
        "--length_penalty",
        type=float,
        default=default_config_infer.get("length_penalty"),
        help="Length penalty",
    )
    decoding_options.add_argument(
        "--max_length",
        type=int,
        default=default_config_infer.get("max_length"),
        help="Maximum decoding length",
    )

    subparsers.add_parser("score", help="Score reference file.")

    args = parser.parse_args()

    inference_args = {
        k: v for k, v in vars(args).items() if k in default_config_infer.keys()
    }

    infer(
        args.ckpt,
        args.src_vocab,
        args.tgt_vocab,
        args.device,
        args.config,
        args.input or sys.stdin,
        args.output or sys.stdout,
        args.ref,
        action=args.cmd,
        **inference_args,
    )


def infer(
    checkpoint,
    src_vocab,
    tgt_vocab,
    device,
    config=None,
    input_file=sys.stdin,
    output_file=sys.stdout,
    reference_file=None,
    action="decode",
    **kwargs,
):
    config = load_config(config).get("infer")
    config = merge_config(config, kwargs)

    set_model_parallel_world_size(1)
    set_model_parallel_rank(0)

    source_vocabulary, _ = load_vocabulary(src_vocab)
    target_vocabulary, target_vocabulary_rev = load_vocabulary(tgt_vocab)

    bos = target_vocabulary[BOS_TOKEN]
    eos = target_vocabulary[EOS_TOKEN]

    checkpoint = torch.load(checkpoint, map_location=device)

    model_config = checkpoint.get("model_config")
    if model_config is None:
        model_config = dict(
            src_vocab_size=len(source_vocabulary),
            tgt_vocab_size=len(target_vocabulary),
            share_embeddings=True,
        )

    model_config["src_vocab_size"] = len(source_vocabulary)
    model_config["tgt_vocab_size"] = len(target_vocabulary)
    model = Transformer.from_config(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    dataset = create_inference_dataset(
        input_file,
        source_vocabulary,
        reference_file,
        target_vocabulary,
        batch_size=config.get("batch_size"),
        device=device,
    )

    with fh_out(output_file) as fh:
        if action == "decode":
            for batch in dataset:
                result = beam_search(
                    model,
                    batch["source"],
                    bos,
                    eos,
                    beam_size=config.get("beam_size"),
                    length_penalty=config.get("length_penalty"),
                    max_length=config.get("max_length"),
                    parallel_output=True,
                )

                for hypotheses in result:
                    tokens = hypotheses[0][1]
                    if tokens and tokens[-1] == eos:
                        tokens.pop(-1)
                    tokens = [target_vocabulary_rev[token_id] for token_id in tokens]
                    print(" ".join(tokens), flush=True, file=fh)
        elif action == "score":
            total_ce, per_example_ce = evaluate(
                model,
                dataset,
                target_vocabulary,
                target_vocabulary_rev,
                device=torch.device(device),
                output_score=True,
            )
            with open(reference_file) as rf:
                for ce, exemple in zip(per_example_ce, rf):
                    fh.write(f"{ce:.4f} ||| {exemple}")


if __name__ == "__main__":
    main()
