# import torch
# import transformers
# import itertools
# import pandas as pd
# import fire
# from tqdm import tqdm
# from tqdm.notebook import tqdm
# tqdm.pandas()


# model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
# tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# def open_file(file):
#     with open(file, 'r') as f:
#         l = [i.strip() for i in f]
#     return l

# def save_file(l, name):
#     with open(name, 'w+') as f:
#         for i in l:
#             f.write(i.strip()+'\n')

# def get_align(src,tgt):
#     # pre-processing
#     sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
#     token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
#     wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
#     ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
#     sub2word_map_src = []
#     for i, word_list in enumerate(token_src):
#         sub2word_map_src += [i for x in word_list]
#     sub2word_map_tgt = []
#     for i, word_list in enumerate(token_tgt):
#         sub2word_map_tgt += [i for x in word_list]

#     # alignment
#     align_layer = 8
#     threshold = 1e-3
#     model.eval()
#     with torch.no_grad():
#         out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
#         out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

#         dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

#         softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
#         softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

#         softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

#     align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
#     align_words = set()
#     for i, j in align_subwords:
#         align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )
    
#     d1, d2 = [], []
#     for i,j in sorted(align_words):
#         d1.append(sent_src[i])
#         d2.append(sent_tgt[j])

#     return d1, d2

# def heuristic(x):
#     src,tgt = x["source"],x["target"]
#     # print(src)
#     # print(tgt)
#     x1,x2 = get_align(src,tgt)
#     l = []
#     for i, j in zip(x1,x2):
#         if x1.count(i)>1 and x2.count(j)<2:
#             l.extend(["｟mrk_fuzzy｠", i])
#         else:
#             l.extend([i])
#     return " ".join(l) 

# def main(src_file, tgt_file, dir):
#     df = pd.DataFrame({'source':open_file(src_file),
#                        "target":open_file(tgt_file)})
#     print("data loaded!")
#     df['new_source'] = df.progress_apply(lambda x: heuristic(x), axis=1)
#     save_file(df.new_source.to_list(), dir+'ja.txt')
#     save_file(df.target.to_list(), dir+'en.txt')
    
# if __name__ =='__main__':
#     fire.Fire(main)

import torch
from transformers import BertTokenizer, BertModel
import itertools
import pandas as pd
from tqdm import tqdm
import fire
tqdm.pandas()


# Inicialización del modelo y el tokenizer
model = BertModel.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def open_file(file_path):
    """Lee el contenido de un archivo y devuelve una lista de líneas."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return lines

def save_file(lines, file_path):
    """Guarda una lista de líneas en un archivo."""
    with open(file_path, 'w+', encoding='utf-8') as f:
        for line in lines:
            f.write(line.strip() + '\n')

def get_align(src, tgt):
    """Obtiene la alineación de palabras entre una frase fuente (src) y una frase objetivo (tgt)."""
    # Preprocesamiento
    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    token_src = [tokenizer.tokenize(word) for word in sent_src]
    token_tgt = [tokenizer.tokenize(word) for word in sent_tgt]

    wid_src = [tokenizer.convert_tokens_to_ids(x) for x in token_src]
    wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

    ids_src = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', truncation=True)['input_ids']
    ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True)['input_ids']

    sub2word_map_src = [i for i, word_list in enumerate(token_src) for _ in word_list]
    sub2word_map_tgt = [i for i, word_list in enumerate(token_tgt) for _ in word_list]

    # Alineación
    align_layer = 8
    threshold = 1e-3
    model.eval()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
        softmax_srctgt = torch.nn.functional.softmax(dot_prod, dim=-1)
        softmax_tgtsrc = torch.nn.functional.softmax(dot_prod, dim=-2)

        softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = {(sub2word_map_src[i], sub2word_map_tgt[j]) for i, j in align_subwords}

    aligned_src = [sent_src[i] for i, _ in sorted(align_words)]
    aligned_tgt = [sent_tgt[j] for _, j in sorted(align_words)]

    return aligned_src, aligned_tgt

def heuristic(row):
    """Aplica una heurística para mejorar la alineación de las frases."""
    src, tgt = row["source"], row["target"]
    aligned_src, aligned_tgt = get_align(src, tgt)
    result = []
    for src_word, tgt_word in zip(aligned_src, aligned_tgt):
        if aligned_src.count(src_word) > 1 and aligned_tgt.count(tgt_word) > 1 and tgt_word in row['tok_true'].split(','):
            result.extend([tgt_word])
    return ",".join(list(set(result)))

def counted(row):
    return len(row['reps_tok'].split(',')) if row['reps_tok']!='' else 0

def counting(src,tgt,tok):
    df = pd.DataFrame({'source': src, 'target': tgt, 'tok_true':tok})
    print("Data loaded!")
    df['reps_tok'] = df.progress_apply(heuristic, axis=1)
    df['c'] = df.progress_apply(counted, axis=1)
    print(df[df.c!=0].shape[0])
    print(df['c'].sum())
    return df[df.c!=0].shape[0], df['c'].sum()

def rep(src_file, tgt_file,tok_file):
    a,b = counting(open_file(src_file),
                     open_file(tgt_file),
                     open_file(tok_file))
    print(a)
    print(b)

def main(src_file, tgt_file,tok_file, output_dir):
    """Función principal para la alineación de frases en archivos fuente y objetivo."""
    df = pd.DataFrame({'source': open_file(src_file), 'target': open_file(tgt_file), 'tok_true':open_file(tok_file)})
    print("Data loaded!")
    df['reps_tok'] = df.progress_apply(heuristic, axis=1)
    df['c'] = df.progress_apply(counted, axis=1)
    print(df['c'].sum())
    save_file(df.reps_tok.to_list(), f'{output_dir}/reps_tok7.txt')

if __name__ == '__main__':
    # fire.Fire(main)
    fire.Fire(rep)


