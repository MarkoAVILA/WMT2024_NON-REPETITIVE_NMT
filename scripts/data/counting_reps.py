import pandas as pd
import fire 
# from parsing import parsing1
# import pyonmttok
import spacy
from tqdm import tqdm
import re

pos_src = spacy.load("ja_core_news_lg")
pos_tgt = spacy.load("en_core_web_lg")

def open_file(file):
    with open(file, "r") as f:
        l = [str(i).strip() for i in f]
    return l

def save(name_x, l):
    with open(name_x, 'w+') as f:
        for i in l:
            f.write(i+'\n')

def count_j(s, pos_nlp):
    pattern = r'<target[^>]*>([^<]*)<\\target>'
    l_pos = ["NOUN","VERB","ADJ"] #"ADP""ADP"
    l_i = []
    if pos_nlp is not None:
        words_pos = []
        token = pos_nlp(" ".join(s))
        for t in token:
            if t.pos_ in l_pos and t.text not in words_pos:
                words_pos.append(t.text)
                # print(words_pos)
        for k in words_pos:
            # print(k.text)
            if s.count(k)>1:
                l_i.append(k)
    else:
            # Find all matches in the input string
            print('source:',s)
            matches = re.findall(pattern, s)
            print("matches:", matches)
            l_i = [i for i in matches]

    print(l_i)
    return l_i 

def count_reps(x,y):
    a, b = [], []
    for i,j in tqdm(zip(x,y)):
        x_i = i.split(' ')
        x_j = j.split(' ')
        a.append(count_j(i, None))
        b.append(count_j(x_j, pos_tgt))
        # b.append(count_j(j, None))
    return a, b

def main(src_file, tgt_file, name_x):
    src, tgt =  open_file(src_file), open_file(tgt_file)
    a, b = count_reps(src, tgt)
    a_ = [",".join(i) for i in a]
    b_ = [",".join(j) for j in b]
    c = [str(i)+'\t'+str(j) for i,j in zip(a_,b_)]
    save(name_x, c)



if __name__ =='__main__':
    fire.Fire(main)
    
    

