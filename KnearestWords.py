from gensim.models import Word2Vec
from scipy import spatial
import numpy as np
import operator

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

def cosine_most_similars (model, word, topn):
    distances = {}
    vec1 = model[word]
    for item in model.keys():
        if item != word:
            vec2 = model[item]
            if (len(vec1) != len(vec2)):
                print("Vetor is not in the same direction \n")
                continue
            # This is the cosine similarity function in Word Similaritys
            cosine_similarity = ((2 - spatial.distance.cosine(vec1, vec2))/2)*4 
            distances[(word, item)] = cosine_similarity 
    sorted_distances = sorted(distances.items(), key=operator.itemgetter(1), reverse=True)
    print('Similarity', sorted_distances[:topn])
    return sorted_distances[:topn]

def KnearestWords(word, topn = 5):
    model = load_mappings('./W2V_150.txt')
    res_array = cosine_most_similars(model, word, topn)
    f = open('./results/Knearest_similarity.txt', 'w',encoding='utf-8')
    f.write('K-nearest Similirity of ' + str(word) + ': \n')
    for res in res_array:
        f.write(str(res) + '\n')


KnearestWords('nguyễn_thị_tươi')
