import argparse
import glob
import os

from itertools import zip_longest

import torch
import sys
# sys.path.append("/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/pytorch-transformer/src/")
# import transformer
from transformer.data import load_vocabulary
from transformer.utils import get_logger, init_logger


def main():
    init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints_dir", help="Directory containing the checkpoints"
    )
    parser.add_argument(
        "--output_checkpoint",
        help="Path to the output checkpoint",
    )

    subparsers = parser.add_subparsers(help="Checkpoint update command", dest="cmd")
    parser_average = subparsers.add_parser("average", help="Average latest checkpoints")
    parser_average.add_argument(
        "--max_to_average",
        type=int,
        default=10,
        help="Maximum number of checkpoints to average",
    )

    parser_update_vocabulary = subparsers.add_parser(
        "update_vocabulary", help="Update checkpoint for vocabulary"
    )
    parser_update_vocabulary.add_argument(
        "--current_checkpoint",
        help="Path to checkpoint to update"
        "(if not specified, latest checkpoint from 'checkpoints_dir' will be used)",
    )
    parser_update_vocabulary.add_argument(
        "--src_vocab",
        help="Old source vocabulary from current checkpoint",
    )
    parser_update_vocabulary.add_argument(
        "--src_vocab_repl",
        help="New source vocabulary for the updated checkpoint",
    )
    parser_update_vocabulary.add_argument(
        "--tgt_vocab",
        help="Old target vocabulary from current checkpoint",
    )
    parser_update_vocabulary.add_argument(
        "--tgt_vocab_repl",
        help="New target vocabulary for the updated checkpoint",
    )

    args = parser.parse_args()

    if args.cmd == "average":
        checkpoints = get_checkpoints(args.checkpoints_dir)[-args.max_to_average :]
        output_path = (
            args.output_checkpoint
            if args.output_checkpoint
            else "averaged_checkpoint.pt"
        )
        average_checkpoints(checkpoints, output_path)
    elif args.cmd == "update_vocabulary":
        current_checkpoint_path = (
            args.current_checkpoint
            if args.current_checkpoint is not None
            else get_latest_checkpoint(args.checkpoints_dir)
        )
        new_checkpoint_path = (
            args.output_checkpoint
            if args.output_checkpoint
            else "checkpoint-updated-vocab.pt"
        )
        update_checkpoint_for_vocab(
            current_checkpoint_path,
            new_checkpoint_path,
            args.src_vocab,
            args.src_vocab_repl,
            args.tgt_vocab,
            args.tgt_vocab_repl,
        )


def get_checkpoint_step(path):
    """Returns the training step from a checkpoint."""
    filename = os.path.basename(path)
    filename, _ = os.path.splitext(filename)
    return int(filename.split("-")[-1])


def get_checkpoints(directory):
    """Returns the list of checkpoints in a directory, ordered on the step number."""
    if not os.path.exists(directory):
        return []

    checkpoints = glob.glob(os.path.join(directory, "checkpoint-*.pt"))
    checkpoints = sorted(checkpoints, key=get_checkpoint_step)
    return checkpoints


def clean_checkpoint_directory(directory, max_to_keep):
    """Removes old checkpoints to keep a maximum number of checkpoints in a directory."""
    checkpoints = get_checkpoints(directory)

    while len(checkpoints) > max_to_keep:
        os.remove(checkpoints.pop(0))


def get_latest_checkpoint(directory):
    """Returns the latest checkpoint in a directory."""
    checkpoints = get_checkpoints(directory)
    return checkpoints[-1] if checkpoints else None


def average_checkpoints(checkpoints, output_path):
    logger = get_logger()

    num_checkpoints = len(checkpoints)
    logger.info("Averaging %d checkpoints", num_checkpoints)

    averaged_model = None

    for checkpoint in checkpoints:
        logger.info("Loading %s", checkpoint)
        checkpoint = torch.load(checkpoint, map_location="cpu")

        model = checkpoint["model"]
        model = {key: value / num_checkpoints for key, value in model.items()}

        if averaged_model is None:
            averaged_model = model
        else:
            for key, value in model.items():
                averaged_model[key] += value

    logger.info("Saving %s", output_path)
    checkpoint = {"model": averaged_model}
    torch.save(checkpoint, output_path)


def update_checkpoint_for_vocab(
    current_checkpoint_path,
    new_checkpoint_path,
    old_src_vocab,
    new_src_vocab,
    old_tgt_vocab,
    new_tgt_vocab,
):
    checkpoint = torch.load(current_checkpoint_path, map_location="cpu")

    update_vocab_in_checkpoint(
        checkpoint,
        old_src_vocab,
        new_src_vocab,
        old_tgt_vocab,
        new_tgt_vocab,
    )

    torch.save(checkpoint, new_checkpoint_path)


def update_vocab_in_checkpoint(
    checkpoint,
    old_src_vocab,
    new_src_vocab,
    old_tgt_vocab,
    new_tgt_vocab,
):
    model_config = checkpoint.get("model_config", {})
    source_vocab_mapping, target_vocab_mapping = None, None
    if new_src_vocab:
        (
            source_vocab_mapping,
            old_source_vocabulary,
            new_source_vocabulary,
        ) = get_vocabulary_mapping(old_src_vocab, new_src_vocab)
    if model_config["share_embeddings"]:
        assert (
            old_src_vocab == old_tgt_vocab
        ), "Previous vocabularies do not match for a model with shared embeddings"
        assert (
            new_src_vocab == new_tgt_vocab
        ), "New vocabularies do not match for a model with shared embeddings"
    if new_tgt_vocab:
        (
            target_vocab_mapping,
            old_target_vocabulary,
            new_target_vocabulary,
        ) = get_vocabulary_mapping(old_tgt_vocab, new_tgt_vocab)

    model_state_dict = checkpoint.get("model")
    optimizer_state = checkpoint.get("optimizer", {}).get("state", {}).items()

    def _map_parameter(mapping, opt_param, param, old_vocab_len):
        param_shape = param.shape
        assert (
            param_shape[0] == old_vocab_len
        ), "Old vocabulary size doesn't match checkpoint."

        def _map_tensor(t):
            new_t = torch.zeros(len(mapping), *param_shape[1:])
            for i, j in enumerate(mapping):
                if j >= 0:
                    new_t[i] = t[j]
            return new_t

        new_param = _map_tensor(param)
        param.data = new_param.data
        if opt_param is not None:
            param_num, opt_param_state = opt_param
            for k, opt_state in opt_param_state.items():
                if k != "step":
                    assert param_shape == opt_state.shape
                    new_opt_state = _map_tensor(opt_state)
                    opt_param_state[k] = new_opt_state

    model_trainable_params = [
        (k, v)
        for k, v in model_state_dict.items()
        if k.find("position_embeddings") == -1
    ]

    old_src_vocab_size = len(old_source_vocabulary)
    model_src_vocab_size = model_config.get("src_vocab_size")
    if model_src_vocab_size:
        assert (
            old_src_vocab_size == model_src_vocab_size
        ), "Previous source vocabulary size doesn't match the model configuration"

    old_tgt_vocab_size = len(old_target_vocabulary)
    model_tgt_vocab_size = model_config.get("tgt_vocab_size")
    if model_tgt_vocab_size:
        assert (
            old_tgt_vocab_size == model_tgt_vocab_size
        ), "Previous target vocabulary size doesn't match the model configuration"

    for model_param, opt_param in zip_longest(model_trainable_params, optimizer_state):
        param_name, param = model_param
        if source_vocab_mapping and "src_embeddings" in param_name:
            _map_parameter(source_vocab_mapping, opt_param, param, old_src_vocab_size)
        if target_vocab_mapping and any(
            n in param_name for n in ("tgt_embeddings", "output_layer")
        ):
            _map_parameter(target_vocab_mapping, opt_param, param, old_tgt_vocab_size)

    model_config["src_vocab_size"] = len(new_source_vocabulary)
    model_config["tgt_vocab_size"] = len(new_target_vocabulary)


def get_vocabulary_mapping(current_vocab_path, new_vocab_path, mode="replace"):
    """Maps vocabulary new indices to old ones. -1 means that the entry is new."""
    mode = mode.lower()
    if mode not in ("merge", "replace"):
        raise ValueError("invalid vocab update mode: %s" % mode)

    current_vocab, _ = load_vocabulary(current_vocab_path)
    new_vocab, new_vocab_list = load_vocabulary(new_vocab_path)

    mapping = []
    if mode == "merge":
        _, final_vocab_list = load_vocabulary(current_vocab_path)
        mapping = list(range(len(current_vocab)))
        for new_word in new_vocab:
            if current_vocab.get(new_word) is None:
                mapping.append(-1)
                final_vocab_list.append(new_word)
    elif mode == "replace":
        final_vocab_list = new_vocab_list
        for new_word in new_vocab:
            idx = current_vocab.get(new_word)
            if idx is not None:
                mapping.append(idx)
            else:
                mapping.append(-1)

    return mapping, current_vocab, new_vocab


if __name__ == "__main__":
    main()
