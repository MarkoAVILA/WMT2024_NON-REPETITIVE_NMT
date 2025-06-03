import fire
import ast

def g(l_n):
    l = []
    for i in l_n:
        l_tmp = []
        for j in i:
            l_tmp.extend(j)
        l_tmp = [str(i) for i in l_tmp]
        l.append(",".join(l_tmp))
    return l

def save(file, l):
    with open(file, 'w+') as f:
        for i in l:
            f.write(str(i).strip()+'\n')

def parsering(PATH, name):
    with open(PATH,'r') as f:
        l = [i.strip() for i in f]
    l_p, l_n = [], []
    l_idx, l_no_idx = [], []
    for idx, i in enumerate(l):
        tupla = ast.literal_eval(i)[1][0]
        palabra = [i[0] for i in tupla]
        number = [i[1][0]for i in tupla]
        if palabra:
            l_idx.append(idx)
            l_p.append(palabra)
            l_n.append(number)
        else:
            l_no_idx.append(idx)
            l_p.append("")
            l_n.append("")
    l_p = [",".join(i) for i in l_p]
    l_n = g(l_n)
    save(str(name)+'.tokens', l_p)
    save(str(name)+'.pos', l_n)


def parsing1(PATH):
    with open(PATH,'r') as f:
        l = [i.strip() for i in f]
    l_p, l_n = [], []
    l_idx, l_no_idx = [], []
    for idx, i in enumerate(l):
        tupla = ast.literal_eval(i)[1][0]
        palabra = [i[0] for i in tupla]
        number = [i[1][0]for i in tupla]
        if palabra:
            l_idx.append(idx)
            l_p.append(palabra)
            l_n.append(number)
        else:
            l_no_idx.append(idx)
            l_p.append("")
            l_n.append("")
    l_p = [",".join(i) for i in l_p]
    l_n = g(l_n)
    return l_p, l_n

# if __name__ == '__main__':
#     fire.Fire(parsering)
#     # fire.Fire(parsing1)