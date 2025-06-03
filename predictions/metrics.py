import sacrebleu
from sacrebleu import corpus_bleu
import fire

def main(pred_file, ref_file, output):
    l_pred, l_ref = [], []
    with open(pred_file, 'r') as f1, open(ref_file, 'r') as f2:
        for i,j in zip(f1,f2):
            l_pred.append(i)
            l_ref.append(j)
    BLEU_ = corpus_bleu(l_pred, [l_ref]).score
    with open(output,'w+') as f:
        f.write(str(BLEU_))

if __name__ == '__main__':
    fire.Fire(main)