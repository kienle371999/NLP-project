import  io
import numpy as np
from scipy import spatial, stats
from sklearn.metrics.pairwise import cosine_similarity

# delation
fSimlex = './Visim-400.txt' # ViCon 

# reading vsimlex
f = open(fSimlex, 'r',encoding='utf-8')
vsl = f.readlines()
f.close()

def load_mappings(path):
    word_mapping = dict()
    with open(path, 'r') as f:
        for line in f.readlines():
            word, vec = line.split(' ', 1)                        
            vec = np.fromstring(vec, sep=' ')                      
            word=word.strip()
            word=word.replace(" ","_")

            word_mapping[word] = vec                               
    return word_mapping


# load model

def Vietnamese_WordSimilarity():
    model = load_mappings('./W2V_150.txt')
    f = open('./Visim-400.txt', 'r')
    rs=[]
    v=[]
    for i in f:
        s=i.split()
        u1 = s[0].strip()
        u2 = s[1].strip()

        if not(u1 in model):
            continue
        if not(u2 in model):
            continue
     
        v1=model[u1.strip()]
        v2=model[u2.strip()]
        if (len(v1)!=len(v2)):
            print(" Vetor is not in the same direction \n")
            continue
     
        k = ((2 - spatial.distance.cosine(v1, v2))/2)*4
        
        v.append(float(s[3].strip())) 
        print(u1 +"- "+u2+ " = "+str(k)+'\n')
        rs.append(k)

    print(rs,"\n")
    print(v,"\n")
    f.close()

    f = open('./results/similarity_result.txt', 'w',encoding="utf-8")
    f.write('Word 1      Word 2       Similarity \n')
    for i in range(len(rs)):
        s = vsl[i].split()
        f.write(s[0]+'  '+s[1] +'        '+str(rs[i])+ '   '+ str(v[i])+'\n')
    f.write('Pearson: ' +  str(stats.pearsonr(rs, v)) + '\n')
    f.write('Spearman: ' + str(stats.spearmanr(rs, v)) + '\n')
    f.close()
    print('done')

Vietnamese_WordSimilarity()
