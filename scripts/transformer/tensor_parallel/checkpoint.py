import torch

from .initialize import get_model_parallel_world_size
from .utils import divide


def compare2tensor(tensor_a, tensor_b):
    if not torch.all(tensor_a.eq(tensor_b)):
        return False
    return True


def gather_object(obj, is_master, world_size):
    cpu_group = torch.distributed.new_group(backend="gloo")
    full_obj = [None for _ in range(world_size)]
    if not is_master:
        torch.distributed.gather_object(obj=obj, group=cpu_group)
    else:
        torch.distributed.gather_object(
            obj=obj, object_gather_list=full_obj, group=cpu_group
        )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return full_obj


def get_parallel_model_info(model, model_state_dict=None):
    info_model = {}
    for name, param in model.module.named_parameters():
        info_model[name] = {
            "tensor_parallel": param.tensor_parallel,
            "partition_dim": param.partition_dim if param.tensor_parallel else 0,
            "stride": param.stride if param.tensor_parallel else 1,
        }
    if model_state_dict is not None:
        for name, param in model_state_dict.items():
            if name not in info_model:
                info_model[name] = {
                    "tensor_parallel": False,
                    "partition_dim": 0,
                    "stride": 1,
                }

    return info_model


def merge_partitions(
    param_layer, parallel_info_layer, partitions_layer_state_dict, world_size, logger
):
    logger.debug(
        f"     2 first partition is SAME: "
        f"{compare2tensor(partitions_layer_state_dict[0], partitions_layer_state_dict[1])}"
    )
    param_size = list(param_layer.size())
    if len(param_size) > 2:
        raise ValueError("Not support with partition_dim > 2")

    if parallel_info_layer["tensor_parallel"]:
        partition_dim = parallel_info_layer["partition_dim"]
        stride = parallel_info_layer["stride"]
        param_size[partition_dim] = param_size[partition_dim] * world_size
        logger.debug(
            f"     merged         type: {param_layer.dtype}, size: {param_size}"
        )
        logger.debug(
            f"     parallel parameter merge with stride {stride} along dimention {partition_dim}"
        )
        param_layer.resize_(param_size)
        if stride == 1:
            with torch.no_grad():
                torch.cat(partitions_layer_state_dict, partition_dim, out=param_layer)
        else:
            per_partition_size = None
            for partition in partitions_layer_state_dict:
                if per_partition_size is None:
                    per_partition_size = partition.size(partition_dim)
                else:
                    assert per_partition_size == partition.size(partition_dim)
            per_partition_per_stride_size = divide(per_partition_size, stride)
            # Chunk and build a list.
            chunks = None
            for i, partition in enumerate(partitions_layer_state_dict):
                chunk = torch.split(
                    partition, per_partition_per_stride_size, dim=partition_dim
                )
                if chunks is None:
                    chunks = [0] * (len(partitions_layer_state_dict) * len(chunk))
                chunks[i :: len(partitions_layer_state_dict)] = chunk
            with torch.no_grad():
                torch.cat(chunks, partition_dim, out=param_layer)
    else:
        logger.debug(
            f"     merged         type: {param_layer.dtype}, size: {param_size}"
        )
        logger.debug("     none-parallel parameter, simple copy from rank 0")


def gather_model_state_dict(model_state_dict, model, is_master, logger):
    # Transfer weight of optimizer to CPU and gather them
    world_size = get_model_parallel_world_size()
    for key, value in model_state_dict.items():
        model_state_dict[key] = value.cpu()

    full_model_state_dict = gather_object(model_state_dict, is_master, world_size)
    if not is_master:
        return {}

    info_model = get_parallel_model_info(model, model_state_dict)

    for key in model_state_dict.keys():
        logger.debug(f" > working on {key} ...")
        partitions_layer_state_dict = []
        for rank in range(world_size):
            partition_layer_state_dict = full_model_state_dict[rank][key]
            partitions_layer_state_dict.append(partition_layer_state_dict)
            logger.debug(
                f"     partition {rank}    type: {partition_layer_state_dict.dtype}, "
                f"size: {list(partition_layer_state_dict.size())}"
            )

        with torch.no_grad():
            tmp_state = model_state_dict[key].clone()
        merge_partitions(
            tmp_state,
            info_model[key],
            partitions_layer_state_dict,
            world_size,
            logger,
        )
        with torch.no_grad():
            model_state_dict[key] = tmp_state
        for i in range(world_size):
            del full_model_state_dict[i][key]

    del full_model_state_dict[:]
    del full_model_state_dict


def load_model_state_dict(checkpoint_model, model, device, offset, world_size):
    for param_name, param in model.module.named_parameters():
        ckpt_t = checkpoint_model[param_name]
        if not param.tensor_parallel:
            with torch.no_grad():
                param.data.copy_(ckpt_t.data)
        else:
            per_partition_per_stride_size = divide(
                param.data.size(param.partition_dim), param.stride
            )
            weight_list = torch.split(
                ckpt_t, per_partition_per_stride_size, dim=param.partition_dim
            )

            rank_weight_list = weight_list[offset::world_size]
            with torch.no_grad():
                torch.cat(rank_weight_list, dim=param.partition_dim, out=param.data)
        del checkpoint_model[param_name]
        model.to(device)

    del checkpoint_model


def gather_optimizer_state_dict(optimizer_state_dict, model, is_master, logger):
    # Transfer weight of optimizer to CPU and gather them
    world_size = get_model_parallel_world_size()
    optimizer_state = optimizer_state_dict["state"]
    for key, value in optimizer_state.items():
        optimizer_state[key] = {
            sub_key: sub_value.cpu() for sub_key, sub_value in value.items()
        }

    full_optim_state_dict = gather_object(optimizer_state, is_master, world_size)
    if not is_master:
        return {}

    info_model = get_parallel_model_info(model)

    for i, (param_name, param_attr) in enumerate(info_model.items()):
        for key, value in optimizer_state[i].items():
            logger.debug(f" > working on {param_name} with key {key} ...")
            if value.numel() == 1:
                logger.debug("     none-parallel parameter, simple copy from rank 0")
                continue

            partitions_layer_state_dict = []
            for rank in range(world_size):
                partition_layer_state_dict = full_optim_state_dict[rank][i][key]
                partitions_layer_state_dict.append(partition_layer_state_dict)
                logger.debug(
                    f"     partition {rank}    type: {partition_layer_state_dict.dtype}, "
                    f"size: {list(partition_layer_state_dict.size())}"
                )
            with torch.no_grad():
                tmp_state = optimizer_state[i][key].clone()
            merge_partitions(
                tmp_state, param_attr, partitions_layer_state_dict, world_size, logger
            )
            with torch.no_grad():
                optimizer_state[i][key] = tmp_state
        for _ in range(world_size):
            del full_optim_state_dict[_][i]

    del full_optim_state_dict[:]
    del full_optim_state_dict


def load_optimizer_state_dict(
    checkpoint_optimizer, optimizer, model, device, offset, world_size
):
    for i, (_, param) in enumerate(model.module.named_parameters()):
        for key, value in checkpoint_optimizer["state"][i].items():
            if value.numel() == 1:
                checkpoint_optimizer["state"][i][key].to(device)
                continue

            if not param.tensor_parallel:
                checkpoint_optimizer["state"][i][key].to(device)
            else:
                per_partition_per_stride_size = divide(
                    param.data.size(param.partition_dim), param.stride
                )
                weight_list = torch.split(
                    value, per_partition_per_stride_size, dim=param.partition_dim
                )

                rank_weight_list = weight_list[offset::world_size]
                with torch.no_grad():
                    res = torch.cat(
                        rank_weight_list, dim=param.partition_dim
                    ).contiguous()
                    checkpoint_optimizer["state"][i][key] = res.to(device)

    optimizer.load_state_dict(checkpoint_optimizer)
    checkpoint_optimizer.clear()
    del checkpoint_optimizer
