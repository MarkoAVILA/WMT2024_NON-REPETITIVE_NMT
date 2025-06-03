BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<blank>"
PAD_ID = 0


def load_vocabulary(path):
    """Loads the vocabulary from a path."""

    with open(path) as vocabulary:
        ids_to_tokens = [line.rstrip("\r\n") for line in vocabulary]

    if UNK_TOKEN not in ids_to_tokens:
        ids_to_tokens.append(UNK_TOKEN)

    tokens_to_ids = {token: i for i, token in enumerate(ids_to_tokens)}

    return tokens_to_ids, ids_to_tokens


def encode_line(
    line, vocabulary, add_bos=False, add_eos=False, tokenize_fn=None, pad_len=None
):
    """Converts a text line into a list of token IDs."""

    bos_id = vocabulary[BOS_TOKEN]
    eos_id = vocabulary[EOS_TOKEN]
    unk_id = vocabulary[UNK_TOKEN]

    line = line.rstrip("\r\n")

    if tokenize_fn is None:
        tokens = line.split()
    else:
        tokens = tokenize_fn(line)

    if not tokens:
        return []

    ids = [vocabulary.get(token, unk_id) for token in tokens]

    if add_bos:
        ids.insert(0, bos_id)
    if add_eos:
        ids.append(eos_id)

    if pad_len:
        padding = [PAD_ID] * (pad_len - len(ids))
        ids = ids + padding

    return ids
