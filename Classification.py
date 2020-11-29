from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import numpy as np

def training_dataset():
    for file in ['./antonym-synonym set/Synonym_vietnamese.txt', './antonym-synonym set/Antonym_vietnamese.txt']:
        f = open(file, 'r')
        j=[0,0]
        array_list = []
        for index, i in enumerate(f):
            if index > 0:
                word1, word2 = i.split(' ', 1)
                array_list.append([word1, word2])
        
        model = Word2Vec(sentences=array_list, min_count = 1, size = 150, window = 5)
        model.save('model_word2vec.bin')
        f.close()

def transform_to_vec(model, word1, word2):
    """
    get two words as input, and transform it to a feature vector
    """
    vec1, vec2 = model[word1], model[word2]
    print("vec1", vec1)
    print("vec2", vec2)
    return np.concatenate((vec1, vec2, vec1*vec2, np.abs(vec1-vec2), vec1+vec2))


def load_mlp_training_data():
    training_dataset()
    model = Word2Vec.load('model_word2vec.bin')
    X = []
    Y = []
    for file in ['./antonym-synonym set/Synonym_vietnamese.txt', './antonym-synonym set/Antonym_vietnamese.txt']:
        f = open(file, 'r')
        j=[0,0]
        for i in f:
            j[0]+=1
            word1, word2 = i.split(' ', 1)
            if word1 in model and word2 in model:
                j[1]+=1
                X.append(transform_to_vec(model, word1, word2))
                if file == './antonym-synonym set/Synonym_vietnamese.txt':
                    Y.append('SYN')
                else:
                    Y.append('ANT')
        f.close()
        print("In %r, %r of %r are in the corpus.\n" %(file, j[1], j[0]))
    X, Y = np.array(X), np.array(Y)
    return X, Y

def testing_dataset():
    for file in ['./ViCon-400/400_noun_pairs.txt', './ViCon-400/400_verb_pairs.txt', './ViCon-400/600_adj_pairs.txt']:
        f = open(file, 'r')
        j=[0,0]
        array_list = []
        for i in f:
            print('line', i)
            word1, word2, relation = i.split()
            array_list.append([word1, word2])
        
        model = Word2Vec(sentences=array_list, min_count = 1, size = 150, window = 5)
        model.save('model_word2vec_test.bin')
        f.close()

def load_mlp_testing_data():
    testing_dataset()
    model = Word2Vec.load('model_word2vec_test.bin')
    X = []
    Y = []
    for file in ['./ViCon-400/400_noun_pairs.txt', './ViCon-400/400_verb_pairs.txt', './ViCon-400/600_adj_pairs.txt']:
        f = open(file, 'r')
        j=[0,0]
        for i in f:
            j[0]+=1
            word1, word2, relation = i.split()
            if word1 in model and word2 in model:
                j[1]+=1
                X.append(transform_to_vec(model, word1, word2))
                Y.append(relation)
                
        f.close()
        print("In %r, %r of %r are in the corpus.\n" %(file, j[1], j[0]))
    X, Y = np.array(X), np.array(Y)
    return X, Y


def training_data():
    X_train, Y_train = load_mlp_training_data()
    print("X_train", X_train.shape)
    print("Y_train", Y_train.shape)

    # classifier
    print('Training...')
    clf = MLPClassifier(max_iter = 1000, hidden_layer_sizes=(300,2), activation='relu')
    clf.fit(X_train, Y_train)
    f= open("./results/training_result.txt", 'w')
    f.write('Training accuracy on antonym-synonym set: ' + str(clf.score(X_train, Y_train)))

    # evaluation on test set

    X_test, Y_test = load_mlp_testing_data()

    print('Accuracy: ', clf.score(X_test, Y_test))

    pred= clf.predict(X_test)
    acc = accuracy_score(Y_test, pred)

    f1 = f1_score(Y_test, pred, average='macro')
    recall = recall_score(Y_test, pred, average='macro')
    precision = precision_score(Y_test, pred, average='macro')

    f1_weighted = f1_score(Y_test, pred, average='weighted')
    recall_weighted = recall_score(Y_test, pred,  average='weighted')
    precision_weighted = precision_score(Y_test, pred,  average='weighted')
 
    print('Accuracy: ', acc,  "\n")
    
    print('F1: ', f1, ' weighted ', f1_weighted, "\n")
    print('recall: ', recall,' weighted ', recall_weighted, "\n")
    print('precision: ', precision,' weighted ', precision_weighted, "\n")
    
    #write 
    f= open("./results/testing_result.txt", 'w')
    f.write('Accuracy: '+ str( acc)+ "\n")

    f.write('recall: '+ str( recall)+"  weighted:"+str(recall_weighted)+ "\n")
    f.write('precision: '+ str( precision)+"  weighted:"+str(precision_weighted)+ "\n")
    f.write('F1: '+ str( f1)+"  weighted:"+str(f1_weighted)+ "\n")
    f.close()

training_data()



