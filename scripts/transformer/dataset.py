"""Classes and functions to build and transform datasets."""

import collections
import itertools
import multiprocessing
import queue
import random
import threading
import time

import torch
import sys
# sys.path.append("/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/pytorch-transformer/src/")
# import transformer
from transformer.data import PAD_ID, encode_line
from transformer.utils import get_logger, init_logger


def create_training_dataset(
    source_dataset,
    target_dataset,
    source_vocabulary,
    target_vocabulary,
    example_weights=None,
    batch_size=4096,
    batch_type="tokens",
    pad_to_multiple=1,
    maximum_source_length=150,
    maximum_target_length=150,
    num_accum_batches=None,
    device="cpu",
    num_epochs=None,
    num_shards=1,
    shard_index=0,
    prefetch_size=None,
    shuffle_buffer_size=None,
    seed=None,
    is_shuffle=True,
    batch_autotune=False,  # TODO : no shuffling, sharding, prefetching, etc
):
    """Creates a dataset with all transformations required for training."""

    if isinstance(source_dataset, str):
        source_dataset = TextFileDataset(source_dataset)
    if isinstance(target_dataset, str):
        target_dataset = TextFileDataset(target_dataset)
    if isinstance(example_weights, str):
        example_weights = TextFileDataset(example_weights, to_float=True)

    datasets = collections.OrderedDict(
        {
            "source": source_dataset,
            "target": target_dataset,
        }
    )
    if example_weights is not None:
        datasets["example_weights"] = example_weights

    dataset = ZipMapDataset(datasets)

    if num_shards > 1:
        dataset = ShardDataset(dataset, num_shards, shard_index)

    if is_shuffle:
        dataset = ShuffleDataset(dataset, shuffle_buffer_size)

    dataset = RepeatDataset(dataset, num_repeats=num_epochs)
    repeat_dataset = dataset

    max_pad_source = maximum_source_length if batch_autotune else None
    max_pad_target = maximum_target_length if batch_autotune else None
    dataset = MapDataset(
        dataset,
        EncodeTokens(
            source_vocabulary, target_vocabulary, max_pad_source, max_pad_target
        ),
    )
    dataset = FilterDataset(
        dataset,
        FilterByLength(maximum_source_length, maximum_target_length),
    )

    if batch_type == "tokens":
        dataset = BatchByTokensDataset(
            dataset,
            batch_size=batch_size,
            length_fn=length_fn,
            length_bucket_width=pad_to_multiple,
            maximum_length=max(maximum_source_length, maximum_target_length),
            drop_remainder=False,
        )
    else:
        dataset = BatchDataset(dataset, batch_size, drop_remainder=False)
    # If the dataset is repeated we should check that there is at least 1 batch
    # per epoch otherwise the iterator can hang infinitely without generating anything.
    dataset = CounterDataset(dataset)
    repeat_dataset.set_counter_dataset(dataset)

    if prefetch_size is None:
        prefetch_size = num_accum_batches if num_accum_batches is not None else 1

    # Prepare batches in a separate process for true parallelism,
    # then bufferize in a separate thread.
    dataset = PrefetchDataset(
        dataset,
        prefetch_size=prefetch_size,
        use_threading=False,
        seed=seed if is_shuffle else None,
    )
    dataset = MapDataset(dataset, ConvertToTensor(device, pad_to_multiple))
    dataset = PrefetchDataset(dataset, prefetch_size=prefetch_size, use_threading=True)

    if num_accum_batches is not None:
        dataset = GroupDataset(dataset, num_accum_batches)

    return dataset


def create_inference_dataset(
    source_path,
    source_vocabulary,
    target_path=None,
    target_vocabulary=None,
    batch_size=30,
    device="cpu",
):
    if source_path is None:
        return None
    datasets = collections.OrderedDict({"source": TextFileDataset(source_path)})

    if target_path is not None:
        datasets["target"] = TextFileDataset(target_path)
    dataset = ZipMapDataset(datasets)
    dataset = MapDataset(dataset, EncodeTokens(source_vocabulary, target_vocabulary))
    dataset = BatchDataset(dataset, batch_size, drop_remainder=False)
    dataset = MapDataset(dataset, ConvertToTensor(device))
    return dataset


class EncodeTokens:
    """Transformation to encode text lines into a list of token IDs."""

    def __init__(
        self,
        source_vocabulary,
        target_vocabulary,
        max_pad_source=None,
        max_pad_target=None,
    ):
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary
        self.max_pad_source = max_pad_source
        self.max_pad_target = max_pad_target

    def __call__(self, element):
        source = element["source"]
        target = element.get("target")

        source = encode_line(
            source, self.source_vocabulary, add_eos=True, pad_len=self.max_pad_source
        )

        if target is None:
            return collections.OrderedDict({"source": source})

        target = encode_line(
            target,
            self.target_vocabulary,
            add_bos=True,
            add_eos=True,
            pad_len=self.max_pad_target,
        )
        if target:
            target_in = target[:-1]
            target_out = target[1:]
        else:
            target_in = []
            target_out = []

        output = collections.OrderedDict(
            {"source": source, "target_in": target_in, "target_out": target_out}
        )
        element.update(output)
        element.pop("target", None)

        return element


class FilterByLength:
    """Filter condition to keep elements satisfying the length constraints."""

    def __init__(self, maximum_source_length, maximum_target_length):
        self.maximum_source_length = maximum_source_length
        self.maximum_target_length = maximum_target_length

    def __call__(self, element):
        source = element["source"]
        target = element["target_in"]
        return (
            0 < len(source) <= self.maximum_source_length
            and 0 < len(target) <= self.maximum_target_length
        )


def length_fn(element):
    """Returns the representative length for a parallel source/target example."""
    source = element["source"]
    target = element["target_in"]
    return max(len(source), len(target))


class ConvertToTensor:
    """Transformation to convert Python lists to PyTorch tensors."""

    def __init__(self, device, pad_to_multiple=1):
        self.device = device
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, elements):
        result = collections.OrderedDict(
            (
                (
                    k,
                    to_tensor(
                        v,
                        device=self.device,
                        pad_to_multiple=self.pad_to_multiple,
                    ),
                )
                if "source" in k or "target" in k
                else (k, torch.tensor(v, device=self.device))
                for k, v in elements.items()
            )
        )
        return result


def to_tensor(batch_ids, device=None, pad_to_multiple=1):
    """Converts a batch of token IDs into a dense 2D tensor."""
    maximum_length = (
        max(len(ids) for ids in batch_ids) if isinstance(batch_ids, list) else 1
    )
    if maximum_length % pad_to_multiple != 0:
        maximum_length += pad_to_multiple - (maximum_length % pad_to_multiple)

    batch_ids = [ids + [PAD_ID] * (maximum_length - len(ids)) for ids in batch_ids]

    return torch.tensor(batch_ids, device=device)


class TextFileDataset:
    """Read lines from a text dataset."""

    def __init__(self, file, to_float=False):
        self._file = file
        self._to_float = to_float

    def __iter__(self):
        if isinstance(self._file, str):
            with open(self._file) as f:
                yield from self._generate_lines(f)
        else:
            yield from self._generate_lines(self._file)

    def _generate_lines(self, file):
        for line in file:
            res = line.rstrip("\r\n")
            if self._to_float:
                res = float(res)
            yield res


class ZipDataset:
    """Read elements from parallel datasets."""

    def __init__(self, *datasets):
        self._datasets = datasets

    def __iter__(self):
        for elements in itertools.zip_longest(*self._datasets):
            yield elements


class ZipMapDataset:
    """Read elements from parallel map-style datasets."""

    def __init__(self, datasets):
        self._datasets = datasets

    def __iter__(self):
        keys = self._datasets.keys()
        for elements in zip(*self._datasets.values()):
            elements = collections.OrderedDict(zip(keys, elements))
            yield elements


class RepeatDataset:
    """Repeat a dataset."""

    def __init__(self, dataset, num_repeats=None):
        self._dataset = dataset
        self._num_repeats = num_repeats
        self._counter_dataset = None

    def set_counter_dataset(self, counter_dataset):
        self._counter_dataset = counter_dataset

    def __iter__(self):
        for epoch in itertools.count(start=1):
            yield from iter(self._dataset)

            if self._num_repeats is not None and epoch == self._num_repeats:
                break

            # No batches produced in one epoch: do not repeat an empty dataset.
            if self._counter_dataset is not None and self._counter_dataset.counter == 0:
                get_logger().warning(
                    "No batches were generated in one epoch. "
                    "Stopping the dataset iterator."
                )
                break


class GroupDataset:
    """Group consecutive dataset elements."""

    def __init__(self, dataset, group_size):
        self._dataset = dataset
        self._group_size = group_size

    def __iter__(self):
        group = []

        for batch in self._dataset:
            group.append(batch)

            if len(group) == self._group_size:
                yield group
                group = []
        if group:
            yield group


class ShardDataset:
    """Read a subset of a dataset."""

    def __init__(self, dataset, num_shards, shard_index):
        self._dataset = dataset
        self._num_shards = num_shards
        self._shard_index = shard_index

    def __iter__(self):
        for i, element in enumerate(self._dataset):
            if i % self._num_shards == self._shard_index:
                yield element


class ShuffleDataset:
    """Read dataset elements in a random order."""

    def __init__(self, dataset, buffer_size=None):
        self._dataset = dataset
        self._buffer_size = buffer_size

    def _shuffle_and_yield(self, elements):
        get_logger().info("Shuffling %d elements", len(elements))
        random.shuffle(elements)
        while elements:
            yield elements.pop()

    def __iter__(self):
        elements = []

        for element in self._dataset:
            elements.append(element)

            if self._buffer_size is not None and len(elements) == self._buffer_size:
                yield from self._shuffle_and_yield(elements)

        if elements:
            yield from self._shuffle_and_yield(elements)


class MapDataset:
    """Apply a transformation on dataset elements."""

    def __init__(self, dataset, map_fn):
        self._dataset = dataset
        self._map_fn = map_fn

    def __iter__(self):
        for element in self._dataset:
            yield self._map_fn(element)


class FilterDataset:
    """Keep dataset elements that satisfy a condition."""

    def __init__(self, dataset, filter_fn):
        self._dataset = dataset
        self._filter_fn = filter_fn

    def __iter__(self):
        for element in self._dataset:
            if self._filter_fn(element):
                yield element


class CounterDataset:
    """Count the number of generated elements."""

    def __init__(self, dataset):
        self._dataset = dataset
        self._counter = 0

    @property
    def counter(self):
        return self._counter

    def __iter__(self):
        for element in self._dataset:
            yield element
            self._counter += 1


class BatchDataset:
    """Batch a dataset by the number of elements."""

    def __init__(self, dataset, batch_size, drop_remainder=False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder

    def __iter__(self):
        batch = []

        for element in self._dataset:
            batch.append(element)

            if len(batch) == self._batch_size:
                yield _batch_elements(batch)
                batch = []

        if not self._drop_remainder and batch:
            yield _batch_elements(batch)


class BatchByTokensDataset:
    """Batch a dataset by the number of tokens."""

    def __init__(
        self,
        dataset,
        batch_size,
        length_fn,
        length_bucket_width,
        maximum_length,
        drop_remainder=False,
    ):
        self._dataset = dataset
        self._length_fn = length_fn
        self._length_bucket_width = length_bucket_width
        self._drop_remainder = drop_remainder

        self._max_length_per_bucket = list(
            range(length_bucket_width, maximum_length + 1, length_bucket_width)
        )
        if self._max_length_per_bucket[-1] != maximum_length:
            self._max_length_per_bucket.append(maximum_length)

        self._batch_size_per_bucket = [
            max(batch_size // max_len, 1) for max_len in self._max_length_per_bucket
        ]

        # Reduce batch to a multiple of 8 to enable NVIDIA Tensor Cores.
        self._batch_size_per_bucket = [
            max(batch_size - batch_size % 8, 1)
            for batch_size in self._batch_size_per_bucket
        ]

    def _get_bucket_id(self, length):
        for i, max_length in enumerate(self._max_length_per_bucket):
            if max_length - self._length_bucket_width < length <= max_length:
                return i

    def __iter__(self):
        buckets = [[] for _ in self._max_length_per_bucket]

        for element in self._dataset:
            length = self._length_fn(element)
            bucket_id = self._get_bucket_id(length)
            bucket = buckets[bucket_id]
            bucket.append(element)

            if len(bucket) == self._batch_size_per_bucket[bucket_id]:
                yield _batch_elements(bucket)
                buckets[bucket_id] = []

        if not self._drop_remainder:
            for bucket in buckets:
                if bucket:
                    yield _batch_elements(bucket)


class PrefetchDataset:
    """Prefetch dataset elements in a background process or thread."""

    def __init__(self, dataset, prefetch_size=1, use_threading=False, seed=None):
        self._dataset = dataset
        self._prefetch_size = prefetch_size
        self._use_threading = use_threading
        self._seed = seed

    def _fetch_elements(self, queue):
        if not self._use_threading:
            init_logger()

            if self._seed is not None:
                random.seed(self._seed)

        for element in self._dataset:
            queue.put(element)
        queue.put(None)

    def __iter__(self):
        if self._use_threading:
            queue_cls = queue.Queue
            worker_cls = threading.Thread
        else:
            context = multiprocessing.get_context("spawn")
            queue_cls = context.Queue
            worker_cls = context.Process

        producer_queue = queue_cls(self._prefetch_size)
        producer = worker_cls(
            target=self._fetch_elements, args=(producer_queue,), daemon=True
        )
        producer.start()

        while True:
            element = producer_queue.get()
            if element is None:
                break
            yield element

        producer.join()


class LatencyDataset:
    """Dataset wrapper to compute the latency to get an element from the dataset."""

    def __init__(self, dataset, ignore_first_n=1):
        self._dataset = dataset
        self._avg_latency_us = 0
        self._num_samples = 0
        self._ignore_first_n = ignore_first_n

    @property
    def average_latency_us(self):
        return self._avg_latency_us

    def __iter__(self):
        iterator = iter(self._dataset)

        while True:
            try:
                start = time.time_ns()
                element = next(iterator)
                end = time.time_ns()

                if self._ignore_first_n > 0:
                    self._ignore_first_n -= 1
                else:
                    latency_us = (end - start) / 1000
                    self._avg_latency_us = (
                        self._avg_latency_us * self._num_samples + latency_us
                    ) / (self._num_samples + 1)
                    self._num_samples += 1

                yield element
            except StopIteration:
                break


def _batch_elements(elements):
    if not elements:
        return elements
    if isinstance(elements[0], tuple):
        return tuple(list(batch) for batch in zip(*elements))
    if isinstance(elements[0], dict):
        return {key: [dct[key] for dct in elements] for key in elements[0].keys()}
    if isinstance(elements[0], list):
        return elements
    raise TypeError("Cannot batch element")
