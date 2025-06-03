import torch


def beam_search(
    model,
    src_ids,
    bos,
    eos,
    beam_size=5,
    length_penalty=1,
    max_length=256,
    parallel_output=True,
):
    with torch.no_grad():
        batch_size = src_ids.shape[0]

        encoder_output, src_mask = model.encode(src_ids)
        encoder_output = repeat_beam(encoder_output, beam_size)
        src_mask = repeat_beam(src_mask, beam_size)

        tgt_ids = torch.full(
            (batch_size, beam_size, 1), bos, dtype=src_ids.dtype, device=src_ids.device
        )

        cum_log_probs = torch.full(
            (batch_size, beam_size), float("-inf"), device=src_ids.device
        )
        cum_log_probs[:, 0] = 0

        finished = [False for _ in range(batch_size)]
        finished_hypotheses = [[] for _ in range(batch_size)]

        kv_cache = {}

        for step in range(max_length):
            tgt_inputs = tgt_ids[:, :, -1].view(-1, 1)

            decoder_output = model.decode(
                tgt_inputs,
                encoder_output,
                src_mask=src_mask,
                kv_cache=kv_cache,
            )

            decoder_output = decoder_output[:, -1]
            logits = model.output_layer(
                decoder_output,
                model.tgt_embeddings.word_embeddings.weight,
                parallel_output,
            )

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            log_probs += cum_log_probs.reshape(-1, 1)

            vocab_size = log_probs.shape[-1]

            cum_log_probs, top_ids = torch.topk(
                log_probs.view(-1, beam_size * vocab_size),
                k=beam_size * 2,
                dim=-1,
            )

            from_beam = top_ids // vocab_size
            top_ids = top_ids % vocab_size

            tgt_ids = index_beam(tgt_ids, from_beam)
            tgt_ids = torch.cat([tgt_ids, top_ids.unsqueeze(-1)], dim=-1)

            for i in range(batch_size):
                if finished[i]:
                    continue

                for k in range(beam_size):
                    last_id = top_ids[i, k]

                    if last_id != eos and step + 1 < max_length:
                        continue

                    hypothesis = tgt_ids[i, k, 1:].tolist()
                    score = cum_log_probs[i, k] / (len(hypothesis) ** length_penalty)

                    finished_hypotheses[i].append((score, hypothesis))

                    # Replace the finished hypothesis by an active candidate.
                    for j in range(beam_size, beam_size * 2):
                        if top_ids[i, j] != eos:
                            tgt_ids[i, k] = tgt_ids[i, j]
                            cum_log_probs[i, k] = cum_log_probs[i, j]
                            from_beam[i, k] = from_beam[i, j]
                            top_ids[i, j] = eos
                            break

                if len(finished_hypotheses[i]) >= beam_size:
                    finished[i] = True
                    finished_hypotheses[i] = sorted(
                        finished_hypotheses[i],
                        key=lambda item: item[0],
                        reverse=True,
                    )

            if all(finished):
                break

            tgt_ids = tgt_ids[:, :beam_size].contiguous()
            cum_log_probs = cum_log_probs[:, :beam_size].contiguous()
            from_beam = from_beam[:, :beam_size].contiguous()

            update_kv_cache(kv_cache, from_beam)

    return finished_hypotheses


def repeat_beam(x, beam_size):
    return x.repeat_interleave(beam_size, dim=0)


def index_beam(x, beam_ids):
    batch_size, beam_size = x.shape[:2]

    batch_offset = torch.arange(batch_size, device=beam_ids.device) * beam_size
    flat_beam_ids = (beam_ids + batch_offset.view(-1, 1)).view(-1)

    flat_x = x.view(x.shape[0] * x.shape[1], *x.shape[2:])
    flat_x = flat_x.index_select(0, flat_beam_ids)

    x = flat_x.view(batch_size, flat_x.shape[0] // batch_size, *x.shape[2:])
    return x


def update_kv_cache(kv_cache, beam_ids):
    batch_size, beam_size = beam_ids.shape
    batch_offset = torch.arange(batch_size, device=beam_ids.device) * beam_size
    flat_beam_ids = (beam_ids + batch_offset.view(-1, 1)).view(-1)

    for name, value in kv_cache.items():
        kv_cache[name] = value.index_select(0, flat_beam_ids)
