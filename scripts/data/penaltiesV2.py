import sys
import logging
import argparse
import pyonmttok
import spacy
from collections import defaultdict
import fire

# nlp_e = spacy.load("en_core_web_lg")
nlp_e = spacy.load("ja_core_news_lg")

def tok2stok(stok, tok, joiner):
    t2s = [[]] * len(tok)
    i_stok = 0
    for i_tok, t in enumerate(tok):
        for j_stok in range(i_stok+1, len(stok)+1):
            if ''.join(stok[i_stok:j_stok]).replace(joiner,'') == t.replace(joiner,''):
                #logging.debug('tok: {} === stok: {}'.format(tok[i_tok], stok[i_stok:j_stok]))
                t2s[i_tok] = list(range(i_stok,j_stok))
                i_stok = j_stok
                break
    assert i_stok == len(stok)
    return t2s

def tgt2src(a, s=None, onmt_tok=None):
    src = onmt_tok(s) if s is not None and onmt_tok is not None else None
    doc = nlp_e(s)
    ali = defaultdict(set)
    for i in a.strip().split():
        if '-' not in a:
            continue
        s, t = i.split('-')
        if src is not None:
            sc = src[int(s)]
        for x in doc:
            if x.text.lower()==sc.lower():
                ali[int(t)].add(x.lemma_.lower())
    return ali

def filter_by_dist(l, max):
    if max == 0: # nothing to filter
        return l
    s = set()
    for i in range(len(l)-1):
        if l[i+1]-l[i] <= max:
            s.add(l[i])
            s.add(l[i+1])
    return sorted(s) #list

def filter_by_src(l, a, s, onmt_tok):
    if len(l) <= 1 or s is None or a is None:
        return l
    ali = tgt2src(a, s, onmt_tok)
    s = set()
    for i in range(len(l)-1):
        set_1 = ali[l[i]]
        set_2 = ali[l[i+1]]
        # if len(set_1.intersection(set_2))==0 and (len(set_1)!=0 and len(set_2)!=0):
        #     s.add(l[i])
        #     s.add(l[i+1])
        # if len(set_1.intersection(set_2))>0 and (len(set_1)!=0 and len(set_2)!=0):
        if (len(set_1) == 0 and len(set_2) == 0) or len(set_1.intersection(set_2)) == 0:
            s.add(l[i])
            s.add(l[i+1])
    return sorted(s)

class Penalty():
    def __init__(self, bpe, voc, pos, max, min, joiner):
        # Tokenization level words
        self.onmt_tok = pyonmttok.Tokenizer(mode="aggressive", joiner_annotate=False) 
        # Tokenization level subwords
        self.onmt_stok = pyonmttok.Tokenizer(mode="aggressive", joiner_annotate=True, preserve_placeholders=True, preserve_segmented_tokens=True, segment_case=True, segment_numbers=True, bpe_model_path=bpe, vocabulary_path=voc)
        self.pos = pos
        self.max = max
        self.min = min
        self.joiner = joiner
        self.tokens_penalized = defaultdict(int)
        self.stats = defaultdict(int)
        # Model POSTAG
        self.nlp = spacy.load("en_core_web_lg",  disable=["parser"]) #fr_core_news_lg

    def occurrences(self, tok):
        t2i = {} 
        for i, t in enumerate(tok):
            if len(t) < self.min: ### filter by length of token
                continue
            if not t.isalpha(): ### filter if contains other than letters (no punct, no numbers)
                continue
            t = t.lower() ### use lowercase
            if t not in t2i:
                t2i[t] = {i} 
            else:
                t2i[t].add(i)
        return t2i
    

    def repetitions(self, t2i, a, s):
        rep = []
        for t, set_i in t2i.items():
            if len(set_i) > 1:
                lset_i = filter_by_src(filter_by_dist(sorted(set_i), self.max), a, s, self.onmt_tok)
                if len(lset_i) > 1:
                    rep.append(lset_i)
        return rep

    def __call__(self, idx, t, a=None, s=None, r=None):
        tok = self.onmt_tok(t)
        t2i = self.occurrences(tok)
        reps = self.repetitions(t2i, a, s)
        o = []
        stok = []
        if len(reps):
            t2p = self.analyse(t) #valid words in t according to their pos
            stok = self.onmt_stok(t)
            t2s = tok2stok(stok, tok, self.joiner)
            logging.debug(f'idx: {idx}')
            logging.debug(f'tok: {tok}')
            logging.debug(f'stok: {stok}')
            logging.debug(f't2s: {t2s}')
            logging.debug(f't2i: {t2i}')
            if r is not None:
                r2i = self.occurrences(self.onmt_tok(r))
                logging.debug(f'r2i: {r2i}')
            logging.debug(f't2p: {t2p}')
            logging.debug(f'reps: {reps}')
            ### penalize repetitions in penalty vector
            for rep in reps:
                logging.debug(f'rep: {rep}')
                token = tok[rep[0]]
                if token not in t2p:
                    logging.debug(f'filter repetition, invalid pos for word {token}: {rep}')
                    continue
                if token in r2i and len(r2i[token]) > 1:
                    logging.debug(f'filter repetition, word {token} repeated in reference')
                    continue
                    
                indexs = []
                self.tokens_penalized[token] += 1
                for i in rep[1:]: ### the first token is not penalised
                    inds = []
                    for j in t2s[i]:
                        inds.append(j)
                    indexs.append(inds)
                    self.tokens_penalized[token] += 1
                    self.stats['penalized_tokens'] += 1
                o.append((token, indexs))
            self.stats['penalized_sentences'] += 1
        self.stats['total_sentences'] += 1
        self.stats['total_tokens'] += len(tok)
        if len(o):
            logging.debug(f'o: {o}')
            return (o, stok)
        return ([], [])


    def analyse(self, t):
        POS = self.pos.split(',')
        POS_1 = POS[:3] #['NOUN', 'VERB', 'ADJ']
        POS_2 = POS[-1] #'ADV'
        t2p = set() #set
        doc = self.nlp(t)
        for token in doc:
            if token.pos_ in POS_1 and token.pos_!='AUX':
                t2p.add(token.text.lower())
            elif token.pos_ == POS_2 and token.text.endswith('ment'):
                t2p.add(token.text.lower())
        return t2p


    def show_stats(self):
        if 'total_sentences' in self.stats and 'total_tokens' in self.stats:
            perc_sents = 100.0 * self.stats['penalized_sentences'] / self.stats['total_sentences']
            perc_toks = 100.0 * self.stats['penalized_tokens'] / self.stats['total_tokens']
            logging.info('total sentences: {} penalized: {} ({}%)'.format(self.stats['total_sentences'], self.stats['penalized_sentences'], perc_sents))
            logging.info('total tokens: {} penalized: {} ({}%)'.format(self.stats['total_tokens'], self.stats['penalized_tokens'], perc_toks))
            for token, freq in sorted(self.tokens_penalized.items(), key=lambda kv: kv[1], reverse=True):
                logging.info(f"{token} {freq}")


def main(tgt, src, ali, ref, max=0, min=0, 
         bpe="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/model_converted/bpe_enja30M-28000.en", 
         voc="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/model_converted/joint_vocab_enja30M-56000.ja_en.v10.converted", 
         pos="VERB,ADJ,NOUN,ADP", joiner="￭", verbose=True):
    """
        This script reads from STDIN text data, applies pyonmttok and computes repetitions.
            Features:
                * tgt: target file
                * src: source file
                * ali: source to target alignments file
                * ref: reference file
                * max: maximum distance of repeated words
                * min: minimum length of tokens to be considered for repetitions
                * bpe: BPE codes for tokenizer
                * voc: transformer vocabulary for tokenizer
                * pos: Comma-separated list of part-of-speeches allowed for repeated words(VERB,ADJ,NOUN,ADV)
                * joiner: joiner char for tokenizer(￭)
                * verbose: verbose output

    """
    logging.basicConfig(format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', level=getattr(logging, 'INFO' if not verbose else 'DEBUG'), filename=None)
    pen = Penalty(bpe, voc, pos,max,min,joiner)
    if src is not None and ali is not None:
        fds = open(src, 'r')
        fda = open(ali, 'r')
    if ref is not None:
        fdr = open(ref, 'r')
    
    with open(tgt, 'r') as fdt:
        for idx, t in enumerate(fdt):
            a, s, r = None, None, None
            if src is not None and ali is not None:
                s = fds.readline()
                a = fda.readline()
                if s=='' or a=='': ### reached end of file
                    break
            if ref is not None:
                r = fdr.readline()
                if r=='': ### reached end of file
                    break

            print((idx, pen(idx, t, a, s, r)))

    if src is not None and ali is not None:
        fds.close()
        fda.close()
    if ref is not None:
        fdr.close()
            
    pen.show_stats()
    

if __name__ == '__main__':
    fire.Fire(main)
