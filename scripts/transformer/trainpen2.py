import argparse
import contextlib
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
# import sys
# sys.path.append("/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/pytorch-transformer/src/")
# import transformer
from transformer.checkpoint import (
    clean_checkpoint_directory,
    get_latest_checkpoint,
    update_vocab_in_checkpoint,
)
from transformer.config import auto_tune_batch_size, load_config
from transformer.data import PAD_ID, load_vocabulary
from transformer.datasetpen import create_inference_dataset, create_training_dataset
from transformer.evaluate import evaluate
from transformer.model import Transformer
from transformer.tensor_parallel.checkpoint import (
    gather_model_state_dict,
    gather_optimizer_state_dict,
    load_model_state_dict,
    load_optimizer_state_dict,
)
from transformer.tensor_parallel.cross_entropy import VocabParallelCrossEntropyLoss
from transformer.tensor_parallel.initialize import (
    get_data_parallel_group,
    get_model_parallel_rank,
    initialize_model_parallel,
)
from transformer.utils import (
    free_ram,
    get_current_ram_used,
    get_logger,
    get_port,
    init_logger,
)
from torch.utils.tensorboard import SummaryWriter

# Some optimizers not support with torch.cuda.amp.GradScaler: SparseAdam, LBFGS, NAdam
# TODO
# When using Tensor parallel mode, it is necessary to handle tensor division
# for optimizers that contain weight. Tested with two optimization types Adam and SGD.
# Further testing with other types of optimization is needed
OPTIMIZERS_WITH_WEIGHT = (
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "ASGD",
    "RAdam",
    "RMSprop",
    "Rprop",
)
OPTIMIZERS_WITHOUT_WEIGHT = "SGD"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Path to the source training file")
    parser.add_argument("--tgt", required=True, help="Path to the target training file")
    parser.add_argument(
        "--src_valid", required=False, help="Path to the source validation file"
    )
    parser.add_argument(
        "--tgt_valid", required=False, help="Path to the target validation file"
    )
    parser.add_argument(
        "--src_vocab", required=True, help="Path to the source vocabulary"
    )
    parser.add_argument("--src_vocab_repl", help="Path to the new source vocabulary")
    parser.add_argument(
        "--tgt_vocab", required=True, help="Path to the target vocabulary"
    )
    parser.add_argument(
        "--bpe_tgt", required=True, help="Path to the bpe model target"
    )
    parser.add_argument("--tgt_vocab_repl", help="Path to the new target vocabulary")
    parser.add_argument("--pos", required=True, help="Path to the position of tokens for penalization")
    parser.add_argument("--alpha", required=True, help="Penalization value")
    parser.add_argument("--batch_size", required=True, help="batch size level tokens")
    parser.add_argument(
        "--save_dir", default="checkpoints/", help="Path to the checkpoint directory"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--tensor_parallel", action="store_true", help="Tensor parallel mode"
    )
    parser.add_argument(
        "--vocab_tensor_parallel",
        action="store_true",
        help="Vocab tensor parallel mode",
    )
    parser.add_argument(
        "--config", help="Configuration file used for building and training the model"
    )
    parser.add_argument(
        "--example_weights",
        required=False,
        help="Path to the file containing the weight (loss multiplier) of each example.",
    )
    args = parser.parse_args()

    multiprocess_train(
        "cuda" if args.num_gpus > 0 and torch.cuda.is_available() else "cpu",
        args.src,
        args.tgt,
        args.save_dir,
        args.src_vocab,
        args.tgt_vocab,
        args.bpe_tgt,
        args.pos,
        args.alpha,
        args.batch_size,
        args.src_vocab_repl,
        args.tgt_vocab_repl,
        args.example_weights,
        args.src_valid,
        args.tgt_valid,
        max(args.num_gpus, 1),
        args.tensor_parallel,
        args.vocab_tensor_parallel,
        args.config,
    )


def multiprocess_train(
    device,
    src,
    tgt,
    save_dir,
    src_vocab,
    tgt_vocab,
    bpe_tgt,
    pos,
    alpha,
    batch_size,
    src_vocab_repl=None,
    tgt_vocab_repl=None,
    example_weights=None,
    src_valid=None,
    tgt_valid=None,
    num_proc=1,
    tensor_parallel=False,
    vocab_tensor_parallel=False,
    config=None,
):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(get_port()))

    smp = torch.multiprocessing.get_context("spawn")
    queue = smp.Queue()

    torch.multiprocessing.spawn(
        train,
        args=(
            device,
            src,
            tgt,
            save_dir,
            src_vocab,
            tgt_vocab,
            bpe_tgt,
            pos,
            alpha,
            batch_size,
            src_vocab_repl,
            tgt_vocab_repl,
            example_weights,
            src_valid,
            tgt_valid,
            num_proc,
            tensor_parallel,
            vocab_tensor_parallel,
            config,
            queue,
        ),
        nprocs=num_proc,
        join=True,
    )

    if queue.empty():
        summary = {}
    else:
        summary = queue.get(block=False)

    return get_latest_checkpoint(save_dir), summary

def p_f(x):
    l = x.split(',')
    return [int(i) if i!='' else i for i in l]

def get_mask(path):
    l = []
    with open(path, 'r') as f:
        for i in f:
            l.append(p_f(str(i).strip()))
    return l

def train(
    rank,
    device,
    source_path,
    target_path,
    save_dir,
    source_vocabulary_path,
    target_vocabulary_path,
    bpe_tgt,
    pos_path,
    alpha,
    bs,
    source_vocabulary_path_repl=None,
    target_vocabulary_path_repl=None,
    example_weights_path=None,
    source_valid_path=None,
    target_valid_path=None,
    num_proc=1,
    tensor_parallel=False,
    vocab_tensor_parallel=False,
    config=None,
    queue=None,
):
    config_init = config
    tensor_parallel_size = num_proc if tensor_parallel else 1
    if not tensor_parallel:
        vocab_tensor_parallel = False

    is_master = rank == 0

    writer = SummaryWriter(os.path.dirname(save_dir)+"/runs")

    if is_master:
        init_logger()

    logger = get_logger()

    use_cuda = device == "cuda"
    device = torch.device(device, rank)

    checkpoint_path = get_latest_checkpoint(save_dir)
    checkpoint = None
    if checkpoint_path is not None:
        logger.info("Restoring checkpoint %s", checkpoint_path)

        map_location = {"cuda:0": "cuda:%d" % rank} if use_cuda else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    checkpoint_config = checkpoint.get("model_config", {}) if checkpoint else {}
    config = load_config(config, checkpoint_config).get("train")

    if is_master:
        logger.info("Initializing torch distributed ...")

    if use_cuda:
        torch.cuda.set_device(device)
        distributed_backend = "nccl"
        enable_mixed_precision = config.get(
            "mixed_precision"
        ) and torch.cuda.get_device_capability() >= (7, 0)
        fused_adam = True
    else:
        distributed_backend = "gloo"
        enable_mixed_precision = False
        fused_adam = False

    # Initialize distributed training.
    torch.distributed.init_process_group(
        distributed_backend, rank=rank, world_size=num_proc
    )

    initialize_model_parallel(tensor_parallel_size, logger)

    seed = config.get("seed")
    # It is necessary to initialize the seed so that the random states are the same via the GPUs
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))

    step = 0
    max_step = config.get("max_step") or float("inf")

    if checkpoint:
        checkpoint_fused = (
            checkpoint["optimizer"]["param_groups"][0].get("fused")
            if "optimizer" in checkpoint
            else False
        )
        if not fused_adam and checkpoint_fused:
            raise ValueError(
                "Cannot load optimizer trained with 'fused' on GPU to CPU."
            )
        fused_adam = checkpoint_fused

        step = int(checkpoint["step"])
        if max_step is not None and step >= max_step:
            logger.info("Training already reached max_step = %d", max_step)
            return

    if source_vocabulary_path_repl or target_vocabulary_path_repl:
        update_vocab_in_checkpoint(
            checkpoint,
            source_vocabulary_path,
            source_vocabulary_path_repl,
            target_vocabulary_path,
            target_vocabulary_path_repl,
        )
        source_vocabulary_path = (
            source_vocabulary_path_repl
            if source_vocabulary_path_repl
            else source_vocabulary_path
        )
        target_vocabulary_path = (
            target_vocabulary_path_repl
            if target_vocabulary_path_repl
            else target_vocabulary_path
        )

    share_embeddings = config.get("share_embeddings")
    source_vocabulary, source_vocabulary_rev = load_vocabulary(source_vocabulary_path)
    src_vocab_size = len(source_vocabulary)
    share_vocab = (
        source_vocabulary_path == target_vocabulary_path
        if share_embeddings is None
        else share_embeddings
    )
    if share_vocab:
        target_vocabulary = source_vocabulary
        target_vocabulary_rev = source_vocabulary_rev
        tgt_vocab_size = src_vocab_size
    else:
        target_vocabulary, target_vocabulary_rev = load_vocabulary(
            target_vocabulary_path
        )
        tgt_vocab_size = len(target_vocabulary)

    batch_size = int(bs) #config.get("batch_size")
    if batch_size is None or batch_size == 0:
        if config.get("batch_type") == "tokens":
            min_batch_size = 256
            max_batch_size = 16384
            min_range = 256
        else:
            min_batch_size = 1
            max_batch_size = 512
            min_range = 16
        batch_size = auto_tune_batch_size(
            source_path,
            target_path,
            source_vocabulary_path,
            target_vocabulary_path,
            config,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            min_range=min_range,
            num_devices=num_proc,
        )
    if config.get("batch_autotune", False):
        accum_steps = 2
    else:
        effective_batch_size = 25*int(bs) #config.get("effective_batch_size")
        accum_steps = (
            effective_batch_size // (batch_size * num_proc // tensor_parallel_size)
            if effective_batch_size is not None
            else 1
        )

    if is_master:
        logger.info("Building model ...")

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        num_layers=config.get("num_layers"),
        num_heads=config.get("num_heads"),
        dim_model=config.get("dim_model"),
        dim_ffn=config.get("dim_ffn"),
        dropout=config.get("dropout"),
        share_embeddings=share_vocab,
        share_target_embeddings=config.get("share_target_embeddings"),
        output_layer_bias=config.get("output_layer_bias"),
        vocab_tensor_parallel=vocab_tensor_parallel,
    )

    if is_master:
        logger.info(
            f"Number of parameters: {sum([p.nelement() for p in model.parameters()])}"
        )

    compile_model = config.get("compile_model")
    if compile_model:
        model = torch.compile(model)
    model.to(device)
    if tensor_parallel:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device] if use_cuda else None,
            process_group=get_data_parallel_group(),
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device] if use_cuda else None, static_graph=True
        )

    trainable_parameters = []
    for parameter in model.module.parameters():
        if parameter.requires_grad:
            if not hasattr(parameter, "tensor_parallel"):
                parameter.tensor_parallel = False
            trainable_parameters.append(parameter)

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=1,  # The learning rate is defined by the scheduler.
        betas=config.get("adam_betas"),
        fused=fused_adam,
    )

    # if tensor_parallel and vocab_tensor_parallel:
    #     ce_loss = VocabParallelCrossEntropyLoss(
    #         label_smoothing=config.get("label_smoothing"),
    #         ignore_index=PAD_ID,
    #         reduction="none",
    #     )
    # else:
    #     ce_loss = torch.nn.CrossEntropyLoss(
    #         label_smoothing=config.get("label_smoothing"),
    #         ignore_index=PAD_ID,
    #         reduction="none",
    #     )
    vocab_size = len(target_vocabulary_rev)
    penalties = get_mask(pos_path)
    ce_loss_penalty = CE_loss_pen(target_vocabulary_rev, vocab_size, penalties,smoothing=config.get("label_smoothing"), alpha=float(alpha),ignore_index=PAD_ID, device=device)

    scaler = torch.cuda.amp.GradScaler(enabled=enable_mixed_precision)

    if is_master and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if checkpoint:
        checkpoint_model = checkpoint.get("model")
        if checkpoint_model:
            if tensor_parallel:
                load_model_state_dict(
                    checkpoint_model,
                    model,
                    device,
                    get_model_parallel_rank(),
                    tensor_parallel_size,
                )
                free_ram()
            else:
                model.module.load_state_dict(checkpoint_model)
            del checkpoint["model"]
            del checkpoint_model

        checkpoint_optimizer = checkpoint.get("optimizer")
        if checkpoint_optimizer:
            if tensor_parallel:
                if optimizer.__class__.__name__ in OPTIMIZERS_WITH_WEIGHT:
                    load_optimizer_state_dict(
                        checkpoint_optimizer,
                        optimizer,
                        model,
                        device,
                        get_model_parallel_rank(),
                        tensor_parallel_size,
                    )
                    free_ram()
                elif optimizer.__class__.__name__ == OPTIMIZERS_WITHOUT_WEIGHT:
                    optimizer.load_state_dict(checkpoint_optimizer)
                else:
                    raise ValueError(
                        f"Not support for optimize {optimizer.__class__.__name__} with Tensor Parallel mode"
                    )
            else:
                optimizer.load_state_dict(checkpoint_optimizer)
            del checkpoint["optimizer"]
            del checkpoint_optimizer

        checkpoint_grad_scaler = checkpoint.get("grad_scaler")
        if checkpoint_grad_scaler:
            scaler.load_state_dict(checkpoint_grad_scaler)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        inv_sqrt_decay(
            config.get("learning_rate"),
            config.get("warmup_steps"),
            config.get("initial_learning_rate"),
        ),
    )

    if checkpoint:
        checkpoint_scheduler = checkpoint.get("lr_scheduler")
        if checkpoint_scheduler:
            scheduler.load_state_dict(checkpoint_scheduler)

    dataset = create_training_dataset(
        source_path,
        target_path,
        source_vocabulary,
        target_vocabulary,
        batch_size=batch_size,
        example_weights=example_weights_path,
        batch_type=config.get("batch_type"),
        pad_to_multiple=8 if compile_model else 1,
        maximum_source_length=config.get("max_source_len"),
        maximum_target_length=config.get("max_target_len"),
        device=device,
        num_epochs=1 if config.get("single_pass") else None,
        num_accum_batches=accum_steps,
        num_shards=num_proc if not tensor_parallel else 1,
        shard_index=rank if not tensor_parallel else 0,
        seed=seed,
        is_shuffle=(not tensor_parallel),
        batch_autotune=config.get("batch_autotune", False),
    )

    dataset_valid = create_inference_dataset(
        source_valid_path,
        source_vocabulary,
        target_valid_path,
        target_vocabulary,
        batch_size=30,
        device=device,
    )

    last_log_time = time.time()
    num_tokens = 0

    if is_master:
        logger.info("Accumulate gradients of %d batches", accum_steps)
        logger.info(
            "Optimize %d parameters",
            sum(parameter.numel() for parameter in trainable_parameters),
        )
    dataset_loss = 0
    for batches in dataset:
        # Compute the global batch size for this training step.
        sample_size = sum(b["target_out"].ne(PAD_ID).sum() for b in batches)
        sample_size = torch.as_tensor(sample_size, dtype=torch.float32, device=device)
        if not tensor_parallel:
            torch.distributed.all_reduce(sample_size, op=torch.distributed.ReduceOp.SUM)

        total_loss = 0

        for i, batch in enumerate(batches):
            indexes = batch['indexes']
            print("indx:", indexes)
            print("idcnf:", indexes.tolist())
            source = batch["source"]
            target_in = batch["target_in"]
            target_out = batch["target_out"]
            example_weights = batch.get("example_weights")

            last_batch = i + 1 == len(batches)

            with torch.autocast(
                device.type,
                dtype=torch.float16 if use_cuda else torch.bfloat16,
                enabled=enable_mixed_precision,
            ):
                logits = model(source, target_in)

                if tensor_parallel and vocab_tensor_parallel:
                    # Tensor Parallel
                    # loss = ce_loss(logits.float(), target_out)
                    loss = ce_loss_penalty.forward(indexes, logits, target_out)
                else:
                    loss = ce_loss_penalty.forward(indexes, logits, target_out)
                    # loss = ce_loss(
                    #     logits.view(-1, logits.shape[-1]), target_out.view(-1)
                    # )
                    if example_weights:
                        example_weights = example_weights.unsqueeze(1)
                        example_weights = example_weights.expand_as(target_out)

                        loss = loss * example_weights.reshape(-1)
                # Multiply by world_size because DDP divides the gradients by world_size.
                loss = loss.sum() * num_proc / sample_size / tensor_parallel_size

            with contextlib.nullcontext() if last_batch else model.no_sync():
                # Only synchronize gradients for the last accumulated batch.
                scaler.scale(loss).backward()

            if tensor_parallel:
                # Tensor Parallel
                num_tokens += source.ne(PAD_ID).sum().item() / tensor_parallel_size
                num_tokens += target_in.ne(PAD_ID).sum().item() / tensor_parallel_size
                total_loss += loss.item() / tensor_parallel_size
            else:
                num_tokens += source.ne(PAD_ID).sum().item()
                num_tokens += target_in.ne(PAD_ID).sum().item()
                total_loss += loss.item() / num_proc

            dataset_loss += total_loss

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        step += 1

        if step % config.get("report_every") == 0:
            # Aggregate the metrics from all ranks and send the result in the master process.
            stats = torch.as_tensor(
                [num_tokens, total_loss], dtype=torch.float32, device=device
            )
            torch.distributed.reduce(stats, dst=0, op=torch.distributed.ReduceOp.SUM)

            if is_master:
                num_tokens, total_loss = stats.tolist()

                current_time = time.time()
                elapsed_time = current_time - last_log_time
                last_log_time = current_time

                logger.info(
                    "step = %d"
                    " ; tokens/s = %d"
                    " ; learning rate = %e"
                    " ; loss = %f",
                    step,
                    int(num_tokens / elapsed_time),
                    scheduler.get_last_lr()[0],
                    total_loss,
                )
                writer.add_scalar("train-loss/steps",total_loss, step)
                writer.add_scalar("learning_rate/steps",scheduler.get_last_lr()[0], step)

            num_tokens = 0

        save_steps = config.get("save_every")
        if (
            save_steps is not None
            and step % config.get("save_every") == 0
            and step != max_step
        ):
            save_checkpoint(
                scaler,
                scheduler,
                model,
                optimizer,
                step,
                save_dir,
                logger,
                config,
                is_master,
                tensor_parallel,
            )

        if dataset_valid is not None and step % config.get("validation_every") == 0:
            model.eval()
            predictions_path = (
                os.path.join(save_dir, "predictions-{}.out".format(step))
                if config.get("save_validation_predictions")
                else None
            )
            valid_loss, BLEU_, reps_sentences, reps_tokens= evaluate(
                model.module,
                bpe_tgt,
                config_init,
                target_vocabulary_path,
                dataset_valid,
                target_vocabulary,
                target_vocabulary_rev,
                device,
                enable_mixed_precision,
                predictions_path=predictions_path,
                tensor_parallel=tensor_parallel,
                vocab_tensor_parallel=vocab_tensor_parallel,
            )
            model.train()
            logger.info("step = %d" " ; validation loss = %f" " ; bleu_score= %f" " ; reps_tok_no_accept= %d" " ; reps_sent_no_accept= %d ", step, valid_loss, BLEU_,reps_tokens, reps_sentences)
            writer.add_scalar("validation-loss/steps",valid_loss, step)
            writer.add_scalar("Score BLEU/steps",BLEU_, step)
            writer.add_scalar("#Reps_tok-No Acceptables/steps",reps_tokens, step)
            writer.add_scalar("#Reps_sentences-No Acceptables/steps", reps_sentences, step)
            # writer.add_scalar("#Reps_tok-Acceptables/steps",reps_tokens_no, step)
            # writer.add_scalar("#Reps_sentences-Acceptables/steps", reps_sentences_no, step)


        if step == max_step:
            break
        writer.close()
        

    save_checkpoint(
        scaler,
        scheduler,
        model,
        optimizer,
        step,
        save_dir,
        logger,
        config,
        is_master,
        tensor_parallel,
    )

    stats = torch.as_tensor(
        [num_tokens, total_loss], dtype=torch.float32, device=device
    )
    torch.distributed.reduce(stats, dst=0, op=torch.distributed.ReduceOp.SUM)

    if is_master:
        num_tokens, total_loss = stats.tolist()

        num_steps = step - checkpoint["step"] if checkpoint is not None else step
        summary = {
            "average_loss": dataset_loss / num_steps,
            "last_learning_rate": scheduler.get_last_lr()[0],
            "last_loss": total_loss,
            "last_step": step,
            "num_steps": num_steps,
        }
        queue.put(summary)
        with open(os.path.join(save_dir, "summary.json"), "w+") as f:
            json.dump(summary, f)

def create_mask(target_rev, indexes, penalty_mask, pred, target, alpha, device):
    print("######INDEXES######")
    print(indexes)
    penalties = [penalty_mask[idx] for idx in indexes]
    print("##### Position penalties #####")
    print(penalties)
    P_mask = torch.zeros_like(pred, device=device)
    for i, seq in enumerate(penalties):
        if seq!=['']:
            for j in seq:
                k = target[i,j]
                print("idx:"+str(indexes[i])+ "subtoks:"+str(target_rev[k]))
                P_mask[i,j,k] = alpha
    return P_mask



class CE_loss_pen:
    def __init__(self, target_rev, vocab_size, penalty_mask, smoothing=0.1, alpha=0.9,ignore_index=0, device='cpu') -> None:
        self.target_rev = target_rev
        self.penalty_mask = penalty_mask
        self.vocab_size = vocab_size
        self.negatives = smoothing/self.vocab_size
        self.positives = (1 - smoothing)+smoothing/self.vocab_size
        self.alpha = alpha
        self.padding_idx = ignore_index
        self.device = device

    def forward(self,indexes, pred, target):
        M_pen = create_mask(self.target_rev, indexes, self.penalty_mask, pred,target, self.alpha, self.device)
        pred = torch.nn.functional.log_softmax(pred, dim=2)
        true_dist = torch.full_like(pred, fill_value=self.negatives,device=self.device)
        true_dist.scatter_(2, target.data.unsqueeze(2), self.positives)
        # Mask padding indices
        mask = (target.view(-1,1) != self.padding_idx)
        fact = (true_dist*M_pen)/(1-true_dist*M_pen)

        # Calculate the loss
        loss = torch.sum((mask)*(-true_dist*(pred)+(true_dist*torch.log(fact))).view(-1, self.vocab_size), dim=1)
        return loss.sum()
    
def inv_sqrt_decay(lr, warmup_steps, initial_lr):
    def _fn(step):
        if step < warmup_steps:
            return initial_lr + (lr - initial_lr) * (step / warmup_steps)
        else:
            return lr * math.sqrt(warmup_steps / step)

    return _fn


def save_checkpoint(
    scaler,
    scheduler,
    model,
    optimizer,
    step,
    save_dir,
    logger,
    config,
    is_master,
    tensor_parallel,
):
    logger.debug("Saving weights with RAM used %f", get_current_ram_used())
    model_state_dict = model.module.state_dict()
    optimizer_state_dict = optimizer.state_dict()

    if tensor_parallel:
        gather_model_state_dict(model_state_dict, model, is_master, logger)
        free_ram()
        logger.debug("Gathering model with RAM used %f", get_current_ram_used())

        if optimizer.__class__.__name__ in OPTIMIZERS_WITH_WEIGHT:
            gather_optimizer_state_dict(optimizer_state_dict, model, is_master, logger)
            logger.debug("Gathering optimizer with RAM used %f", get_current_ram_used())
            free_ram()
        elif optimizer.__class__.__name__ == OPTIMIZERS_WITHOUT_WEIGHT:
            # Do nothing
            pass
        else:
            raise ValueError(
                f"Not support for optimize {optimizer.__class__.__name__} with TP mode"
            )

        logger.debug("Gathered done with RAM used %f", get_current_ram_used())

    if is_master:
        checkpoint = {
            "grad_scaler": scaler.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "model": model_state_dict,
            "model_config": config,
            "optimizer": optimizer_state_dict,
            "step": step,
        }

        save_path = os.path.join(save_dir, "checkpoint-%d.pt" % step)
        logger.info("Saving checkpoint %s", save_path)
        torch.save(checkpoint, save_path)
        clean_checkpoint_directory(save_dir, config.get("keep_checkpoints"))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model_state_dict.clear()
    del model_state_dict

    optimizer_state_dict["state"].clear()
    optimizer_state_dict.clear()
    del optimizer_state_dict
    free_ram()

    logger.debug("Saved weights done with RAM used %f", get_current_ram_used())


if __name__ == "__main__":
    main()
