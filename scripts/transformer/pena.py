import sys
# sys.path.append("/nfs/RESEARCH/crego/projects/Repetitions/scripts/")
sys.path.append("/nfs/RESEARCH/avila/NMT/REPETITIONS/")
from penalties import *

def counting(path_src, l_t, path_ali, max=10):
    P = Penalty("/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/model_converted/bpe_enja30M-28000.en", 
                "/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/model_converted/joint_vocab_enja30M-56000.ja_en.v10.converted", 
                lex="/nfs/RESEARCH/crego/projects/GramErrCorr/GEC/resources/Morphalou3.1_CSV.csv", 
                pos='VERB,ADJ,NOUN,ADP', 
                max=max, joiner='￭')
    
    if path_src is not None and path_ali is not None:
        fds = open(path_src, 'r')
        fda = open(path_ali, 'r')
    
    for idx, t in enumerate(l_t):
        if path_src is not None and path_ali is not None:
            s = fds.readline()
            a = fda.readline()
            if s=='' or a=='': ### reached end of file
                break
        else:
            a, s = None, None
        P(t, a, s)

# def counting_new(path_src, l_t, path_ali, max=10):
#     P = Penalty('/nfs/RESEARCH/avila/TRANSFORMER/NEWS_EXP/pytorch-transformer/Models_Systran_converted/joint-bpe32k_2022.en_fr', 
#                 '/nfs/RESEARCH/avila/TRANSFORMER/NEWS_EXP/pytorch-transformer/Models_Systran_converted/joint-vocab-2022_enfr-34k.en_fr.v10.converted', 
#                 lex='/nfs/RESEARCH/crego/projects/GEC/resources/Morphalou3.1_CSV.csv', 
#                 pos='VERB,ADJ,NOUN,ADV', 
#                 max=max, joiner='￭')
    
#     if path_src is not None and path_ali is not None:
#         fds = open(path_src, 'r')
#         fda = open(path_ali, 'r')
    
#     for idx, t in enumerate(l_t):
#         if path_src is not None and path_ali is not None:
#             s = fds.readline()
#             a = fda.readline()
#             if s=='' or a=='': ### reached end of file
#                 break
#         else:
#             a, s = None, None
#         P(t, a, s)



    # with open(path_tgt, 'r') as fdt:
    #     for idx, t in enumerate(fdt):
    #         if path_src is not None and path_ali is not None:
    #             s = fds.readline()
    #             a = fda.readline()
    #             if s=='' or a=='': ### reached end of file
    #                 break
    #         else:
    #             a, s = None, None
    #         P(t, a, s)
    return P.stats['penalized_sentences'], P.stats['penalized_tokens']

if __name__ == "__main__":

    # type='ADJ'
    # fin='truematch' #
    # fin="falsematch"
    # path_src = f"DATASETS/test/source/{type}.{fin}"
    # n=2013
    # path_src = f"/nfs/RESEARCH/avila/TRANSFORMER/NEWS_EXP/newstest{n}.en"
    path_src_1 = f"DATASETS/test/en_1"
    path_src_2 = f"DATASETS/test/en_2"
    # path_tgt = "CORPUS/valid.fr"#"/nfs/RESEARCH/avila/TRANSFORMER/pred_Q1.txt" #"corpus_synthetic/test.fr"
    # path_tgt = f"DATASETS/test/pred/pen_no_equi/{type}.{fin}"
    # n=10
    # path_tgt= f"/nfs/RESEARCH/avila/TRANSFORMER/NEWS_EXP/RANDOM_SAMPLING/beam_5/pred_topk_{n}.txt"
    # path_tgt = f"DATASETS/test/fr.pen_mask"
    # path_tgt = f"/nfs/RESEARCH/avila/TRANSFORMER/NEWS_EXP/RANDOM_SAMPLING/news_test/newstest-total"
    # path_src = "DATASETS/test/en_2"
    # path_tgt = "DATASETS/test/nllb-pred-2.txt"
    # path_tgt = "DATASETS/test/pred-gpt_1.txt"
    # path_tgt = "DATASETS/test/pred-gpt-2_2.txt"
    # path_ali = "/nfs/RESEARCH/crego/projects/Repetitions/raw_epoch_syn/test.fr.gdfa"
    # path_tgt = f"/nfs/RESEARCH/avila/TRANSFORMER/NEWS_EXP/RANDOM_SAMPLING/news_test/newstest{n}.topk_10"
    path_tgt = f"DATASETS/test/deepl.fr"
    path_ali = None
    with open(path_tgt, 'r') as f:
        l_t = [i.strip() for i in f]

    a,b = counting(path_src_1, l_t[:101], path_ali, max=0)
    print(a)
    print(b)
    a,b = counting(path_src_2, l_t[101:], path_ali, max=0)
    print(a)
    print(b)
    # a,b = counting(path_src, l_t, path_ali, max=0)
    # print(a)
    # print(b)
    # Sans alignement
    # True Target
    # #tokens_rep = 2260
    # #sentences_rep = 1515

    # Predicted Target
    # #tokens_rep = 2061
    # #sentences_rep = 1287



