import copy
import json
import os
import subprocess
import sys
import tempfile
import sys
# sys.path.append("/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/pytorch-transformer/src/")
# import transformer
from transformer.utils import get_logger, get_port

default_config = {
    "train": {
        "num_layers": 6,
        "num_heads": 16,
        "dim_model": 1024,
        "dim_ffn": 4096,
        "dropout": 0.1,
        # None means the embeddings are automatically shared if the source and target
        # vocabularies are the same.
        "share_embeddings": None,
        "share_target_embeddings": True,
        "output_layer_bias": False,
        "max_source_len": 150,
        "max_target_len": 150,
        "batch_type": "tokens",
        "batch_size": 16000,
        "effective_batch_size": 400000,
        "label_smoothing": 0.1,
        # Learning rate schedule: inverse square root
        "learning_rate": 0.001,
        "warmup_steps": 4000,
        "initial_learning_rate": 1e-7,
        "adam_betas": (0.9, 0.98),
        "report_every": 20,
        "save_every": 500,
        "validation_every": 500,
        "save_validation_predictions": True,
        "max_step": 50000,
        "keep_checkpoints": 10,
        "compile_model": False,
        "mixed_precision": True,
        "seed": 1234,
    },
    "infer": {
        "batch_size": 16,
        "beam_size": 5,
        "length_penalty": 1,
        "max_length": 256,
    },
}


def merge_config(a, b):
    for key, b_value in b.items():
        if not isinstance(b_value, dict):
            a[key] = b_value
        else:
            a_value = a.get(key)
            if a_value is not None and isinstance(a_value, dict):
                merge_config(a_value, b_value)
            else:
                a[key] = b_value
    return a


def load_config(config=None, checkpoint_config=None):
    if config is None:
        config = {}
    elif isinstance(config, str):
        with open(config) as cf:
            config = json.load(cf)

    # For PN9 compatibility.
    if "options" in config:
        config = config["options"]["config"]

    checkpoint_config = {"train": checkpoint_config}

    merged_config = merge_config(default_config, checkpoint_config)
    merged_config = merge_config(merged_config, config)

    return merged_config


def auto_tune_batch_size(
    source_path,
    target_path,
    source_vocabulary_path,
    target_vocabulary_path,
    config,
    min_batch_size,
    max_batch_size,
    min_range,
    sample_iterations=5,
    num_devices=1,
    scaling_factor=0.7,
    timeout=15 * 60,
):
    logger = get_logger()

    logger.info(
        "Searching the largest batch size between %d and %d with a precision of %d...",
        min_batch_size,
        max_batch_size,
        min_range,
    )

    absolute_min_batch_size = min_batch_size
    stderr_data = None

    while max_batch_size - min_batch_size > min_range:
        batch_size = (max_batch_size + min_batch_size) // 2

        with tempfile.TemporaryDirectory() as tmpdir:
            run_train_config = copy.deepcopy(config)
            run_train_config["batch_autotune"] = True
            run_train_config["batch_size"] = batch_size
            run_train_config["save_every"] = None
            run_train_config["max_step"] = sample_iterations

            run_config = {"train": run_train_config}

            config_path = os.path.join(tmpdir, "batch_size_autotuner.json")
            with open(config_path, mode="w") as config_file:
                json.dump(run_config, config_file)

            env = os.environ.copy()
            env["LOG_LEVEL"] = "ERROR"
            env["MASTER_PORT"] = str(get_port())

            args = [
                "transformer-train",
                "--src",
                source_path,
                "--tgt",
                target_path,
                "--src_vocab",
                source_vocabulary_path,
                "--tgt_vocab",
                target_vocabulary_path,
                "--config",
                config_path,
                "--save_dir",
                tmpdir,
                "--num_gpus",
                str(num_devices),
            ]

            logger.info("Trying training with batch size %d...", batch_size)
            with open(os.devnull, "w") as devnull:
                process = subprocess.Popen(
                    args,
                    stdout=devnull,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                try:
                    _, stderr_data = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.info("... failed (timeout).")
                    max_batch_size = batch_size - 1
                else:
                    if process.returncode != 0:
                        logger.info("... failed.")
                        max_batch_size = batch_size - 1
                    else:
                        logger.info(
                            "... succeeded, continue until the search range is smaller than %d.",
                            min_range,
                        )
                        min_batch_size = batch_size

    if min_batch_size == absolute_min_batch_size:
        if stderr_data is not None:
            logger.error(
                'Last training attempt exited with an error:\n\n"""\n%s"""\n'
                % stderr_data.decode("utf-8")
            )
        raise RuntimeError(
            "Batch size autotuning failed: all training attempts exited with an error "
            "(see last error above). Either there is not enough memory to train this "
            "model, or unexpected errors occured. Please try to set a fixed batch size "
            "in the training configuration."
        )

    batch_size = max(int(scaling_factor * min_batch_size), absolute_min_batch_size)
    logger.info("Batch size auto tuned to %d.", batch_size)
    return batch_size
