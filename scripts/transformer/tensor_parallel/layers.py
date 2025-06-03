import torch

from .initialize import (
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)
from .utils import divide, vocab_range_from_global_vocab_size


def _reduce(input_):
    if get_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_model_parallel_group())

    return input_


def _split(input_):
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        return input_

    # Split along last dimension.
    last_dim = input_.dim() - 1
    last_dim_size = divide(input_.size()[last_dim], world_size)
    input_list = torch.split(input_, last_dim_size, dim=last_dim)

    # Note: torch.split does not create contiguous tensors by default.
    rank = torch.distributed.get_rank(group=get_model_parallel_group())
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        return input_

    group = get_model_parallel_group()
    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _GatherFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def gather_from_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def parallel_logits(input_, word_embeddings_weight, bias=None, parallel_output=True):
    # Parallel logits.
    input_parallel = copy_to_model_parallel_region(input_)
    # Matrix multiply.
    if bias is None:
        logits_parallel = torch.nn.functional.linear(
            input_parallel, word_embeddings_weight
        )
    else:
        logits_parallel = torch.nn.functional.linear(
            input_parallel, word_embeddings_weight, bias
        )
    if parallel_output:
        return logits_parallel

    return gather_from_model_parallel_region(logits_parallel)


def initialize_affine_weight(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride,
):
    weight.tensor_parallel = True
    weight.partition_dim = partition_dim
    weight.stride = stride

    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        return

    # Initialize master weight
    master_weight = torch.empty(
        output_size, input_size, dtype=weight.dtype, requires_grad=False
    )
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(
        master_weight, per_partition_per_stride_size, dim=partition_dim
    )

    rank = get_model_parallel_rank()
    rank_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(rank_weight_list, dim=partition_dim, out=weight)


class VocabParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, pad_id, init_method, stride=1):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = pad_id

        # Divide the weight matrix along the vocaburaly dimension.
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = vocab_range_from_global_vocab_size(
            self.num_embeddings,
            get_model_parallel_rank(),
            get_model_parallel_world_size(),
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        self.weight = torch.nn.parameter.Parameter(
            torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim)
        )
        initialize_affine_weight(
            self.weight,
            self.num_embeddings,
            self.embedding_dim,
            self.num_embeddings_per_partition,
            partition_dim=0,
            init_method=init_method,
            stride=stride,
        )

    def forward(self, input_):
        input_mask = (input_ < self.vocab_start_index) | (
            input_ >= self.vocab_end_index
        )
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        output_parallel = torch.nn.functional.embedding(
            masked_input, self.weight, self.padding_idx
        )
        output_parallel[input_mask, :] = 0.0
        output = reduce_from_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, init_method, stride=1, bias=True):
        super().__init__()

        world_size = get_model_parallel_world_size()
        output_size_per_partition = divide(output_size, world_size)

        self.weight = torch.nn.parameter.Parameter(
            torch.Tensor(output_size_per_partition, input_size)
        )
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.Tensor(output_size_per_partition)
            )
            self.bias.tensor_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        initialize_affine_weight(
            self.weight,
            output_size,
            input_size,
            output_size_per_partition,
            partition_dim=0,
            init_method=init_method,
            stride=stride,
        )

    def forward(self, x):
        x = copy_to_model_parallel_region(x)
        y = torch.nn.functional.linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, init_method, stride=1, bias=True):
        super().__init__()

        world_size = get_model_parallel_world_size()
        input_size_per_partition = divide(input_size, world_size)
        self.weight = torch.nn.parameter.Parameter(
            torch.Tensor(output_size, input_size_per_partition)
        )
        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.Tensor(output_size))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        initialize_affine_weight(
            self.weight,
            output_size,
            input_size,
            input_size_per_partition,
            partition_dim=1,
            init_method=init_method,
            stride=stride,
        )

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight)
        x = reduce_from_model_parallel_region(x)

        if self.bias is not None:
            y = x + self.bias
        else:
            y = x
        return y
