import torch
# import sys
# sys.path.append("/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/pytorch-transformer/src/")
# import transformer
from transformer.data import PAD_ID
from transformer.tensor_parallel import layers
from transformer.tensor_parallel.initialize import get_model_parallel_world_size
from transformer.tensor_parallel.utils import divide


class Transformer(torch.nn.Module):
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        num_layers=6,
        num_heads=16,
        dim_model=1024,
        dim_ffn=4096,
        dropout=0.1,
        max_length=1024,
        share_embeddings=False,
        share_target_embeddings=True,
        output_layer_bias=False,
        vocab_tensor_parallel=False,
        **kwargs,
    ):
        super().__init__()

        self.config = dict(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_model=dim_model,
            dim_ffn=dim_ffn,
            dropout=dropout,
            max_length=max_length,
            share_embeddings=share_embeddings,
            share_target_embeddings=share_target_embeddings,
            output_layer_bias=output_layer_bias,
        )

        self.src_embeddings = TransformerEmbedding(
            dim_model,
            src_vocab_size,
            max_length,
            dropout,
            init_weight_embedding(),
            vocab_tensor_parallel,
        )

        if share_embeddings:
            self.tgt_embeddings = self.src_embeddings
        else:
            self.tgt_embeddings = TransformerEmbedding(
                dim_model,
                tgt_vocab_size,
                max_length,
                dropout,
                init_weight_embedding(),
                vocab_tensor_parallel,
            )

        self.encoder = TransformerEncoder(
            num_layers,
            dim_model,
            dim_ffn,
            num_heads,
            dropout,
            init_weight_linear(),
        )
        self.decoder = TransformerDecoder(
            num_layers,
            dim_model,
            dim_ffn,
            num_heads,
            dropout,
            init_weight_linear(),
        )
        self.output_layer = TransformerHead(
            dim_model, tgt_vocab_size, share_target_embeddings, bias=output_layer_bias
        )
        triangular_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            max_length
        )
        self.register_buffer("triangular_mask", triangular_mask, persistent=False)

    def forward(self, src_ids, tgt_ids):
        encoder_output, src_mask = self.encode(src_ids)
        decoder_output = self.decode(tgt_ids, encoder_output, src_mask=src_mask)
        logits = self.output_layer(
            decoder_output, self.tgt_embeddings.word_embeddings.weight
        )
        return logits

    def encode(self, src_ids):
        src_inputs = self.src_embeddings(src_ids, 0, src_ids.shape[1])

        src_padding_mask = src_ids.eq(PAD_ID)
        src_mask = src_inputs.new_zeros(src_padding_mask.shape)
        src_mask = src_mask.masked_fill(src_padding_mask, float("-inf"))
        src_mask = src_mask.view(-1, 1, 1, src_ids.shape[1])

        memory = self.encoder(src_inputs, mask=src_mask)
        return memory, src_mask

    def decode(self, tgt_ids, encoder_output, src_mask=None, kv_cache=None):
        offset = 0 if kv_cache is None else get_cached_length(kv_cache)

        batch_size, tgt_max_len = tgt_ids.shape
        tgt_real_len = offset + tgt_max_len

        tgt_inputs = self.tgt_embeddings(tgt_ids, offset, tgt_real_len)

        tgt_padding_mask = tgt_ids.eq(PAD_ID).unsqueeze(1)
        tgt_mask = self.triangular_mask[offset:tgt_real_len, :tgt_real_len].unsqueeze(0)
        tgt_mask = tgt_mask.expand(batch_size, -1, -1)
        tgt_mask = tgt_mask.masked_fill(tgt_padding_mask, float("-inf"))
        tgt_mask = tgt_mask.view(-1, 1, tgt_max_len, tgt_real_len)

        outputs = self.decoder(
            tgt_inputs,
            encoder_output,
            mask=tgt_mask,
            memory_mask=src_mask,
            kv_cache=kv_cache,
        )

        return outputs


class TransformerHead(torch.nn.Module):
    def __init__(self, dim_model, vocab_size, share_target_embeddings, bias=False):
        super().__init__()
        self.share_target_embeddings = share_target_embeddings
        self.bias = None
        if not share_target_embeddings:
            self.weight = torch.nn.parameter.Parameter(
                torch.Tensor(vocab_size, dim_model)
            )
            self.weight.tensor_parallel = True
            self.weight.partition_dim = 0
            self.weight.stride = 1
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(vocab_size))
            self.bias.tensor_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = 1

    def forward(self, hidden_states, word_embeddings_weight, parallel_output=True):
        output = layers.parallel_logits(
            hidden_states,
            word_embeddings_weight if self.share_target_embeddings else self.weight,
            bias=self.bias,
            parallel_output=parallel_output,
        )
        return output


class TransformerEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim_model,
        vocab_size,
        max_length,
        embedding_dropout_prob,
        init_method,
        vocab_tensor_parallel,
    ):
        super().__init__()

        self.emb_scale = dim_model**0.5

        if vocab_tensor_parallel:
            self.word_embeddings = layers.VocabParallelEmbedding(
                vocab_size, dim_model, PAD_ID, init_method
            )
        else:
            self.word_embeddings = torch.nn.Embedding(
                vocab_size, dim_model, padding_idx=PAD_ID
            )
            init_method(self.word_embeddings.weight)
        position_embeddings = get_positional_embeddings(max_length, dim_model)
        self.register_buffer("position_embeddings", position_embeddings)
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def forward(self, input_ids, start, end):
        words_embeddings = self.word_embeddings(input_ids) * self.emb_scale
        position_embeddings = self.position_embeddings[start:end].unsqueeze(0)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.embedding_dropout(embeddings)

        return embeddings


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        ffn_dim,
        attention_heads,
        dropout,
        init_method,
        bias=True,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim,
                    ffn_dim,
                    attention_heads,
                    dropout,
                    init_method,
                    bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)
        return x


class TransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        ffn_dim,
        attention_heads,
        dropout,
        init_method,
        bias=True,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embed_dim,
                    ffn_dim,
                    attention_heads,
                    dropout,
                    init_method,
                    i,
                    bias,
                )
                for i in range(num_layers)
            ]
        )

        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x, memory, mask=None, memory_mask=None, kv_cache=None):
        for layer in self.layers:
            x = layer(x, memory, mask=mask, memory_mask=memory_mask, kv_cache=kv_cache)

        x = self.norm(x)
        return x


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        attention_heads,
        dropout,
        init_method,
        bias,
    ):
        super().__init__()

        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadAttention(
            embed_dim,
            attention_heads,
            dropout,
            self_attention=True,
            init_method=init_method,
            bias=bias,
        )

        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout, init_method, bias)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        y = self.self_attention(self.norm1(x), mask=mask)
        x = self.dropout(y) + x

        y = self.ffn(self.norm2(x))
        x = self.dropout(y) + x

        return x


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        attention_heads,
        dropout,
        init_method,
        layer_index,
        bias,
    ):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadAttention(
            embed_dim,
            attention_heads,
            dropout,
            self_attention=True,
            init_method=init_method,
            layer_index=layer_index,
            bias=bias,
        )

        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(
            embed_dim,
            attention_heads,
            dropout,
            self_attention=False,
            init_method=init_method,
            layer_index=layer_index,
            bias=bias,
        )

        self.norm3 = torch.nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout, init_method, bias)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, memory, mask=None, memory_mask=None, kv_cache=None):
        y = self.self_attention(self.norm1(x), mask=mask, kv_cache=kv_cache)
        x = self.dropout(y) + x

        y = self.attention(self.norm2(x), memory, mask=memory_mask, kv_cache=kv_cache)
        x = self.dropout(y) + x

        y = self.ffn(self.norm3(x))
        x = self.dropout(y) + x

        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim,
        attention_heads,
        dropout,
        self_attention,
        init_method,
        layer_index=0,
        bias=True,
    ):
        super().__init__()

        self.world_size = get_model_parallel_world_size()
        self.embed_dim = embed_dim
        self.attention_heads = divide(attention_heads, self.world_size)
        self.hidden_size_per_attention_head = divide(embed_dim, attention_heads)
        self.dropout = dropout

        if self_attention:
            self.in_proj = layers.ColumnParallelLinear(
                embed_dim, embed_dim * 3, init_method, stride=3, bias=bias
            )
        else:
            self.query_proj = layers.ColumnParallelLinear(
                embed_dim, embed_dim, init_method, bias=bias
            )
            self.value_proj = layers.ColumnParallelLinear(
                embed_dim, embed_dim * 2, init_method, stride=2, bias=bias
            )

        self.out_proj = layers.RowParallelLinear(
            embed_dim, embed_dim, init_method, bias=bias
        )

        self.cache_prefix = "self_attention" if self_attention else "attention"
        self.cache_prefix = "%s_%d" % (self.cache_prefix, layer_index)

    def forward(self, query, value=None, mask=None, kv_cache=None):
        if kv_cache is not None:
            cached_key = kv_cache.get("%s_key" % self.cache_prefix)
            cached_value = kv_cache.get("%s_value" % self.cache_prefix)
        else:
            cached_key, cached_value = None, None

        if value is None:
            proj = self.in_proj(query)
            proj = split_heads(proj, self.attention_heads * 3)
            query, key, value = proj.split(self.attention_heads, dim=1)

            if cached_key is not None:
                key = torch.cat([cached_key, key], dim=2)
                value = torch.cat([cached_value, value], dim=2)

        else:
            query = self.query_proj(query)
            query = split_heads(query, self.attention_heads)

            if cached_key is not None:
                key = cached_key
                value = cached_value
            else:
                proj = self.value_proj(value)
                proj = split_heads(proj, self.attention_heads * 2)
                key, value = proj.split(self.attention_heads, dim=1)

        if kv_cache is not None:
            kv_cache["%s_key" % self.cache_prefix] = key
            kv_cache["%s_value" % self.cache_prefix] = value

        # TODO Reimplement scaled_dot_product_attention
        # With Tensor Parallel (TP) mode, during the training process,
        # configuring the dropout layer with a non-zero probability (for example, 0.1)
        # may drop different units between GPUs.
        # That means the model when training with TP will be a little different
        # from when training in normal mode. However, with a small probability,
        # the difference is insignificant. We can improve by synchronizing
        # dropout by random number generator state
        # https://pytorch.org/docs/stable/_modules/torch/random.html#get_rng_state

        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0,
        )

        output = combine_heads(output)
        output = self.out_proj(output)

        return output


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, outer_dim, inner_dim, dropout, init_method, bias=True):
        super().__init__()

        self.inner = layers.ColumnParallelLinear(
            outer_dim, inner_dim, init_method, bias=bias
        )
        self.outer = layers.RowParallelLinear(
            inner_dim, outer_dim, init_method, bias=bias
        )

        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.inner(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.outer(x)
        return x


def init_weight_linear():
    def init_(tensor):
        return torch.nn.init.xavier_uniform_(tensor)

    return init_


def init_weight_embedding():
    def init_(tensor):
        return torch.nn.init.uniform_(tensor, -0.07, 0.07)

    return init_


def split_heads(x, heads):
    x = x.reshape(x.shape[0], x.shape[1], heads, -1)
    x = x.transpose(1, 2)
    return x


def combine_heads(x):
    x = x.transpose(1, 2)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    return x


def get_positional_embeddings(length, depth, device=None):
    channels = torch.arange(depth // 2).unsqueeze(0)
    positions = torch.arange(0, length).unsqueeze(1)
    scaled_positions = positions / torch.pow(10000, (2 * channels) / depth)
    sin = torch.sin(scaled_positions)
    cos = torch.cos(scaled_positions)
    encodings = torch.hstack([sin, cos])
    return encodings.to(device)


def get_cached_length(kv_cache):
    for key, value in kv_cache.items():
        if "self_attention" in key:
            return value.shape[2]
    return 0
