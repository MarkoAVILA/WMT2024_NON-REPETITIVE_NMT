from parsing import parsing1
import pandas as pd

def path2list(file):
    with open(file, 'r') as f:
        l = [i.strip() for i in f]
    return l

def save(l, name):
    with open(name,'w+') as f:
        for i in l:
            f.write(i.strip()+'\n')

def main(dir, percent, penalties1="/nfs/RESEARCH/crego/projects/Repetitions/wmt/en2.syn.penalty_nosrc",
         penalties2="/nfs/RESEARCH/crego/projects/Repetitions/wmt/en2.syn.penalty",
         src='/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/corpus/train_jiji/ja.syn.txt',
         ref="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/corpus/train_jiji/en.txt",
         hyp="/nfs/RESEARCH/avila/WMT2024_NON-REPETITIVE/corpus/train_jiji/en.syn.txt",
         ):
     l_p,l_n = parsing1(penalties1)
     l_p2, l_n2 = parsing1(penalties2)
     df = pd.DataFrame({"source":path2list(src), 
                        "target":path2list(ref), 
                        "target_syn":path2list(hyp), 
                        "pos":l_n, 
                        "token":l_p,
                        "pos2":l_n2,
                        "tokens2":l_p2}
                        )
     
     df_pen = df[(df.pos!="")&(df.pos2=="")]
     print(df_pen.shape[0])
     df_no_pen = df[(df.pos=="") & (df.pos2=="")]
     print(df_no_pen.shape[0])
     df_no_pen = df_no_pen.sample(n=int(df_pen.shape[0]*(100-percent)/percent))
     df_final = pd.concat([df_pen, df_no_pen], axis=0)
     save(df_final.source.to_list(), dir+'ja.txt')
     save(df_final.target_syn.to_list(), dir+'en.txt')
     save(df_final.pos.to_list(), dir+'pos')
     save(df_final.pos.to_list(), dir+'tokens')

if __name__ == "__main__":
    ## filtered
    # main("DATA/filter/d1/", 10) #d1
    # main("DATA/filter/d2/", 30) #d2
    # main("DATA/filter/d3/", 50) #d3
    ## no filter
    main("corpus/d1/", 30) #d1
    main("corpus/d2/", 40) #d2
    main("corpus/d3/", 50) #d3




