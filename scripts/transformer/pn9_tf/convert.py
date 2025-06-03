import argparse
import json
import os
import shutil

from collections import OrderedDict

import numpy as np
import torch

from opennmt.utils.checkpoint import (
    _variables_to_structure,
    get_checkpoint_variables,
    get_step_from_checkpoint_prefix,
)
import sys
sys.path.append("/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/pytorch-transformer/src/")
import transformer
from transformer.config import default_config
from transformer.data import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from transformer.model import get_positional_embeddings

transformer_configs = dict.fromkeys(
    ("TransformerBig", "TransformerBigLB"),
    {
        "num_layers": 6,
        "num_heads": 16,
        "dim_model": 1024,
        "dim_ffn": 4096,
        "dropout": 0.1,
        "max_step": None,
        "batch_size": 0,
    },
)
transformer_configs["TransformerBig"]["effective_batch_size"] = 80000
transformer_configs["TransformerBigLB"]["effective_batch_size"] = 400000
transformer_configs["TransformerBigLB"]["report_every"] = 10

transformer_configs["Transformer"] = {
    "num_layers": 6,
    "num_heads": 8,
    "dim_model": 512,
    "dim_ffn": 2048,
    "dropout": 0.1,
    "max_step": None,
    "batch_size": 0,
    "effective_batch_size": 80000,
}


def convert_vocabulary(vocab_file, converted_vocab_file):
    with open(vocab_file) as vocab, open(converted_vocab_file, "w") as converted_vocab:
        header = True
        header_len = 0
        for index, line in enumerate(vocab):
            # The vocabulary file might start with some comments prefixed with '#'.
            if header and line[0] == "#":
                header_len += 1
                continue
            header = False
            line = line.rstrip("\n\r")
            fields = line.split(" ")
            if len(fields) == 1:
                # No frequency value, the line is just the token.
                token = fields[0]
            else:
                # The code below checks the last field is a frequency and not a part of
                # a badly formatted token.
                try:
                    float(fields[-1])
                    fields.pop()
                except ValueError:
                    pass
                token = " ".join(fields)

            if index - header_len == 0:
                converted_vocab.write(f"{PAD_TOKEN}\n")
                converted_vocab.write(f"{BOS_TOKEN}\n")
                converted_vocab.write(f"{EOS_TOKEN}\n")
            converted_vocab.write("%s\n" % token)
    return converted_vocab_file


def get_attributes(var):
    variable = var
    optimizer_m, optimizer_v = None, None

    if isinstance(var, dict):
        variable = var.get(".ATTRIBUTES", {}).get("VARIABLE_VALUE")
        optimizer_slot = var.get(".OPTIMIZER_SLOT", {}).get("optimizer")
        optimizer_m = (
            optimizer_slot["m"][".ATTRIBUTES"]["VARIABLE_VALUE"]
            if optimizer_slot
            else var.get("Adam")
        )
        optimizer_v = (
            optimizer_slot["v"][".ATTRIBUTES"]["VARIABLE_VALUE"]
            if optimizer_slot
            else var.get("Adam_1")
        )

    variable_value = (
        torch.as_tensor(variable).squeeze() if variable is not None else None
    )
    optimizer_m_value = (
        torch.as_tensor(optimizer_m).squeeze() if optimizer_m is not None else None
    )
    optimizer_v_value = (
        torch.as_tensor(optimizer_v).squeeze() if optimizer_v is not None else None
    )

    return variable_value, optimizer_m_value, optimizer_v_value


def convert_norm_layer(norm_layer_vars, norm_layer_name):
    res = OrderedDict()
    res[f"{norm_layer_name}.weight"] = get_attributes(norm_layer_vars["gamma"])
    res[f"{norm_layer_name}.bias"] = get_attributes(norm_layer_vars["beta"])
    return res


def convert_linear_layer(linear_layer_vars, linear_layer_name=""):
    res = OrderedDict()
    if linear_layer_name:
        linear_layer_name += "."
    kernel = linear_layer_vars.get("kernel")
    if kernel is not None:
        res[f"{linear_layer_name}weight"] = []
        weights = get_attributes(kernel)
        for w in weights:
            if w is not None:
                w = w.transpose(0, 1)
            res[f"{linear_layer_name}weight"].append(w)
    bias = linear_layer_vars.get("bias")
    if bias is not None:
        res[f"{linear_layer_name}bias"] = get_attributes(bias)
    return res


def convert_attn_proj(attn_proj_vars, is_self_attn):
    def _partial_weights(weights, num_splits, index):
        def split_func(w):
            w = w.squeeze()
            w = np.split(w, num_splits, axis=0 if w.ndim == 1 else 1)[index]
            return w

        result = {}
        for k, v in weights.items():
            if isinstance(v, dict):
                split_v = {kk: split_func(vv) for kk, vv in v.items()}
            else:
                split_v = split_func(v)
            result[k] = split_v

        return result

    res = OrderedDict()

    if is_self_attn:
        keys_vars = attn_proj_vars.get("layer", {}).get(
            "linear_keys"
        ) or _partial_weights(attn_proj_vars.get("conv1d"), 3, 1)
        keys = convert_linear_layer(keys_vars)

        values_vars = attn_proj_vars.get("layer", {}).get(
            "linear_values"
        ) or _partial_weights(attn_proj_vars.get("conv1d"), 3, 2)
        values = convert_linear_layer(values_vars)

        queries_vars = attn_proj_vars.get("layer", {}).get(
            "linear_queries"
        ) or _partial_weights(attn_proj_vars.get("conv1d"), 3, 0)
        queries = convert_linear_layer(queries_vars)

        for var in ("weight", "bias"):
            res[f"in_proj.{var}"] = []
            for q, k, v in zip(queries[var], keys[var], values[var]):
                if q is None:
                    assert k is None and v is None
                    res[f"in_proj.{var}"].append(None)
                else:
                    res[f"in_proj.{var}"].append(torch.cat([q, k, v], dim=0))

        linear_output = attn_proj_vars.get("layer", {}).get(
            "linear_output"
        ) or attn_proj_vars.get("conv1d_1")
    else:
        keys_vars = attn_proj_vars.get("layer", {}).get(
            "linear_keys"
        ) or _partial_weights(attn_proj_vars.get("conv1d_1"), 2, 0)
        keys = convert_linear_layer(keys_vars)

        values_vars = attn_proj_vars.get("layer", {}).get(
            "linear_values"
        ) or _partial_weights(attn_proj_vars.get("conv1d_1"), 2, 1)
        values = convert_linear_layer(values_vars)

        queries_vars = attn_proj_vars.get("layer", {}).get(
            "linear_queries"
        ) or attn_proj_vars.get("conv1d")
        res.update(convert_linear_layer(queries_vars, "query_proj"))
        for var in ("weight", "bias"):
            res[f"value_proj.{var}"] = []
            for k, v in zip(keys[var], values[var]):
                if k is None:
                    assert v is None
                    res[f"value_proj.{var}"].append(None)
                else:
                    res[f"value_proj.{var}"].append(torch.cat([k, v], dim=0))

        linear_output = attn_proj_vars.get("layer", {}).get(
            "linear_output"
        ) or attn_proj_vars.get("conv1d_2")

    res.update(convert_linear_layer(linear_output, "out_proj"))
    return res


def convert_attn(attn_vars, attn_type):
    res = OrderedDict()

    is_self_attn = attn_type == "self_attention"
    if not is_self_attn and "0" in attn_vars:
        attn_vars = attn_vars["0"]

    norm = "norm1" if is_self_attn else "norm2"
    layer_norm = attn_vars.get("input_layer_norm") or attn_vars.get("LayerNorm")
    res.update(convert_norm_layer(layer_norm, norm))

    attn_proj_vars = convert_attn_proj(attn_vars, is_self_attn=is_self_attn)
    for k, v in attn_proj_vars.items():
        res[f"{attn_type}.{k}"] = v

    return res


def convert_ffn(ffn_vars, layer_name):
    res = OrderedDict()

    layer_norm = ffn_vars.get("input_layer_norm") or ffn_vars.get("LayerNorm")
    res.update(convert_norm_layer(layer_norm, layer_name))

    linear_layer_vars = {}
    linear_layer_vars["inner"] = ffn_vars.get("layer", {}).get("inner") or ffn_vars.get(
        "conv1d"
    )
    linear_layer_vars["outer"] = ffn_vars.get("layer", {}).get("outer") or ffn_vars.get(
        "conv1d_1"
    )

    for linear_name, linear_vars in linear_layer_vars.items():
        for k, v in convert_linear_layer(linear_vars, linear_name).items():
            res[f"ffn.{k}"] = v

    return res


def convert_encoder(encoder_vars):
    res = OrderedDict()

    layer_prefix = "layer_"
    layers = encoder_vars.get("layers") or {
        int(k.replace(layer_prefix, "")): v
        for k, v in encoder_vars.items()
        if k.startswith(layer_prefix)
    }
    for l_idx, l_value in layers.items():
        attn = l_value.get("self_attention") or l_value.get("multi_head")
        layer_res = convert_attn(attn, "self_attention")
        layer_res.update(convert_ffn(l_value["ffn"], "norm2"))
        for k, v in layer_res.items():
            res[f"encoder.layers.{l_idx}.{k}"] = v

    layer_norm = encoder_vars.get("layer_norm") or encoder_vars.get("LayerNorm")
    for k, v in convert_norm_layer(layer_norm, "norm").items():
        res[f"encoder.{k}"] = v

    return res


def convert_decoder(decoder_vars):
    res = OrderedDict()

    layer_prefix = "layer_"
    layers = decoder_vars.get("layers") or {
        int(k.replace(layer_prefix, "")): v
        for k, v in decoder_vars.items()
        if k.startswith(layer_prefix)
    }
    for l_idx, l_value in layers.items():
        self_attn = l_value.get("self_attention") or l_value.get("masked_multi_head")
        layer_res = convert_attn(self_attn, "self_attention")

        attn = l_value.get("attention") or l_value.get("multi_head")
        layer_res.update(convert_attn(attn, "attention"))

        layer_res.update(convert_ffn(l_value["ffn"], "norm3"))
        for k, v in layer_res.items():
            res[f"decoder.layers.{l_idx}.{k}"] = v

    layer_norm = decoder_vars.get("layer_norm") or decoder_vars.get("LayerNorm")
    for k, v in convert_norm_layer(layer_norm, "norm").items():
        res[f"decoder.{k}"] = v

    return res


def convert_embeddings(embedding_vars, config):
    res = OrderedDict()
    res["src_embeddings.word_embeddings.weight"] = get_attributes(
        embedding_vars["features_inputter"]["embedding"]
    )

    labels_inputter = embedding_vars.get("labels_inputter")
    if labels_inputter:
        res["tgt_embeddings.word_embeddings.weight"] = get_attributes(
            embedding_vars["labels_inputter"]["embedding"]
        )
        if config:
            config["train"]["share_embeddings"] = False
    else:
        res["tgt_embeddings.word_embeddings.weight"] = (
            res["src_embeddings.word_embeddings.weight"][0],
        )
        if config:
            config["train"]["share_embeddings"] = True

    return res


def convert_model(model_tf_ckpt, config=None):
    output = OrderedDict()

    # Add the optimizer parameters in the same order as they are added to the Transformer model.
    embedding_layer = model_tf_ckpt.get("examples_inputter")
    if not embedding_layer:
        embedding_layer = {
            "features_inputter": {"embedding": model_tf_ckpt["encoder"].get("w_embs")}
        }
        tgt_emb = model_tf_ckpt["decoder"].get("w_embs")
        if tgt_emb is not None:
            embedding_layer["labels_inputter"] = {"embedding": tgt_emb}
    output.update(convert_embeddings(embedding_layer, config))

    output.update(convert_encoder(model_tf_ckpt["encoder"]))

    decoder = model_tf_ckpt["decoder"]
    output.update(convert_decoder(decoder))

    # Output layer should be added after inserting embeddings.
    output_layer = decoder.get("output_layer") or decoder.get("dense")
    output.update(convert_linear_layer(output_layer, "output_layer"))
    output_weight = output.get("output_layer.weight")
    if not output_weight:
        if config:
            config["train"]["share_target_embeddings"] = True
    else:
        if config:
            config["train"]["share_target_embeddings"] = False

    if config:
        config["train"]["output_layer_bias"] = (
            True if output.get("output_layer.bias") else False
        )

    # max_length is currently hard-coded for positional embeddings, but we might make it an option.
    if config:
        output["src_embeddings.position_embeddings"] = (
            get_positional_embeddings(1024, config["train"]["dim_model"]),
        )

        output["tgt_embeddings.position_embeddings"] = (
            get_positional_embeddings(1024, config["train"]["dim_model"]),
        )

    return output


def convert_optimizer_params(optimizer_tf_ckpt, output_config, step=None):
    if step:
        step = step.item()

    opt_iter = optimizer_tf_ckpt.get("iter")
    if opt_iter:
        step = get_attributes(opt_iter)[0].item()

    assert step, f"Step {step} is undefined"

    beta_1 = optimizer_tf_ckpt.get("beta_1") or optimizer_tf_ckpt.get("beta1_power")
    beta_2 = optimizer_tf_ckpt.get("beta_2") or optimizer_tf_ckpt.get("beta2_power")
    if isinstance(beta_1, dict):
        beta_1 = get_attributes(beta_1)[0].item()
    if isinstance(beta_2, dict):
        beta_2 = get_attributes(beta_2)[0].item()

    adam_betas = None
    if beta_1 and beta_2:
        adam_betas = (beta_1, beta_2)
        output_config["train"]["adam_betas"] = adam_betas

    params = {
        # The learning rate is defined by the scheduler.
        "lr": 1,
        # Add the deffault parameters, necessary to restore the checkpoint.
        "eps": 1e-08,
    }

    if adam_betas:
        params["betas"] = adam_betas
    else:
        params["betas"] = default_config["train"]["adam_betas"]

    decay = optimizer_tf_ckpt.get("decay", 0)
    if decay:
        decay = get_attributes(decay)[0].item()
    params["weight_decay"] = decay

    scheduler = {
        "last_epoch": step,
        # Add the deffault parameters, necessary to restore the checkpoint.
        "lr_lambdas": [None],
    }
    return params, scheduler, step


def convert(model_path, output_path, no_optimizer=False, convert_vocab=False):
    # Copy all files.
    shutil.copytree(
        model_path,
        output_path,
        ignore=shutil.ignore_patterns(
            "*ckpt*", "*checkpoint*", "config.json", "checksum.md5", "manifest.json"
        ),
    )

    config = os.path.join(model_path, "config.json")
    with open(config) as cf:
        config = json.load(cf)

    # Convert vocabularies
    if convert_vocab:
        src_vocab = config.get("vocabulary", {}).get("source", {}).get("path")
        if not src_vocab:
            src_vocab = (
                config.get("tokenization", {}).get("source", {}).get("vocabulary")
            )
        if not src_vocab:
            raise (ValueError("Couldn't find source vocabulary in the configuration"))
        src_vocab_basename = os.path.basename(src_vocab)
        convert_vocabulary(
            os.path.join(model_path, src_vocab_basename),
            os.path.join(output_path, src_vocab_basename + ".converted"),
        )

        tgt_vocab = config.get("vocabulary", {}).get("target", {}).get("path")
        if not tgt_vocab:
            tgt_vocab = (
                config.get("tokenization", {}).get("target", {}).get("vocabulary")
            )
        if not tgt_vocab:
            raise (ValueError("Couldn't find target vocabulary in the configuration"))
        tgt_vocab_basename = os.path.basename(tgt_vocab)
        if src_vocab_basename != tgt_vocab_basename:
            convert_vocabulary(
                os.path.join(model_path, tgt_vocab_basename),
                os.path.join(output_path, tgt_vocab_basename + ".converted"),
            )

    # Retrieve model options
    options = config["options"]

    model_type = options.pop("model_type", None) or options.pop("model", None)
    if "transformer_big_lb" in model_type:
        model_type = "TransformerBigLB"
    if model_type in transformer_configs:
        output_config = {"train": transformer_configs[model_type]}
    else:
        raise (ValueError(f"Unknown model type {model_type}"))

    # Add length limits
    src_max_len = (
        options.get("config", {}).get("train", {}).get("maximum_features_length")
    )
    if src_max_len:
        output_config["train"]["max_source_len"] = src_max_len

    tgt_max_len = (
        options.get("config", {}).get("train", {}).get("maximum_labels_length")
    )
    if tgt_max_len:
        output_config["train"]["max_target_len"] = tgt_max_len

    chkpt = get_checkpoint_variables(model_path)
    chkpt_structure = _variables_to_structure(chkpt)

    model_chkpt_structure = chkpt_structure.get(
        "model", chkpt_structure.get("transformer")
    )
    converted_chkpt = convert_model(model_chkpt_structure, output_config)

    model_state_dict = {k: v[0] for k, v in converted_chkpt.items()}

    optimizer_chkpt_structure = chkpt_structure.get("optimizer") or chkpt_structure.get(
        "optim", {}
    )
    optimizer_params, scheduler_params, step = convert_optimizer_params(
        optimizer_chkpt_structure, output_config, chkpt_structure.get("global_step")
    )

    # For V1-style checkpoints
    optimizer_chkpt_transformer_structure = optimizer_chkpt_structure.get("transformer")
    if optimizer_chkpt_transformer_structure:
        converted_optimizer_chkpt = convert_model(optimizer_chkpt_transformer_structure)
    else:
        converted_optimizer_chkpt = converted_chkpt
    optimizer_state = {
        idx: {"step": step, "exp_avg": v[1], "exp_avg_sq": v[2]}
        for idx, (k, v) in enumerate(converted_optimizer_chkpt.items())
        if len(v) > 1
    }

    optimizer_params["params"] = list(optimizer_state.keys())
    optimizer_state_dict = {
        "state": optimizer_state,
        "param_groups": [optimizer_params],
    }

    checkpoint = {
        "model": model_state_dict,
        "model_config": output_config["train"],
        "step": step,
    }
    if not no_optimizer:
        checkpoint["optimizer"] = optimizer_state_dict
        checkpoint["lr_scheduler"] = scheduler_params

    save_path = os.path.join(output_path, "checkpoint-%d.pt" % step)
    torch.save(checkpoint, save_path)

    config["options"]["config"] = output_config
    config["framework"] = "torch"

    with open(os.path.join(output_path, "config.json"), "w") as cf:
        json.dump(config, cf, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to PN9 TF model")
    parser.add_argument(
        "--output_model_path",
        required=True,
        help="Path to converted Pytorch transformer model path",
    )
    parser.add_argument("--no_optimizer", action="store_true")
    parser.add_argument("--convert_vocab", action="store_true")

    args = parser.parse_args()

    convert(
        args.model_path, args.output_model_path, args.no_optimizer, args.convert_vocab
    )


if __name__ == "__main__":
    main()
