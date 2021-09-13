import json
import random

import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec


class ncClassifier(object):

    def __init__(self, vectors, clf):
        self.embeddings = vectors
        self.clf = TopKRanker(clf)  # here clf is LR
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def read_node_label_downstream(filename):
        """ may be used in node classification task;
            part of labels for training clf and
            the result served as ground truth;
            note: similar method can be found in graph.py -> read_node_label
        """
        fin = open(filename, 'r')
        X = []
        Y = []
        while 1:
            line = fin.readline()
            if line == '':
                break
            vec = line.strip().split(' ')
            X.append(int(vec[0]))
            Y.append(vec[1:])
        fin.close()
        return X, Y

    def split_train_evaluate(self, X, Y, train_precent, seed=0):
        state = np.random.get_state()
        training_size = int(train_precent * len(X))
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))]
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y)
        np.random.set_state(state)
        return self.evaluate(X_test, Y_test)

    def train(self, X, Y, Y_all):
        # to support multi-labels, fit means dict mapping {orig cat: binarized vec}
        self.binarizer.fit(Y_all)
        X_train = [self.embeddings[x] for x in X]
        # since we have use Y_all fitted, then we simply transform
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def predict(self, X, top_k_list):
        X_ = np.asarray([self.embeddings[x] for x in X])
        # see TopKRanker(OneVsRestClassifier)
        # the top k probs to be output...
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def evaluate(self, X, Y):
        # multi-labels, diff len of labels of each node
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)  # pred val of X_test i.e. Y_pred
        Y = self.binarizer.transform(Y)  # true val i.e. Y_test
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        # print(results)
        return results


class TopKRanker(OneVsRestClassifier):  # orignal LR or SVM is for binary clf
    def predict(self, X, top_k_list):  # re-define predict func of OneVsRestClassifier
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[
                probs_.argsort()[-k:]].tolist()  # denote labels
            probs_[:] = 0  # reset probs_ to all 0
            probs_[labels] = 1  # reset probs_ to 1 if labels denoted...
            all_labels.append(probs_)
        return np.asarray(all_labels)


# ------------------link prediction task---------------------------
class lpClassifier(object):

    def __init__(self, vectors):
        self.embeddings = vectors

    # clf here is simply a similarity/distance metric
    def evaluate(self, X_test, Y_test, seed=0):
        test_size = len(X_test)
        Y_true = [int(i) for i in Y_test]
        Y_probs = []
        for i in range(test_size):
            start_node_emb = np.array(
                self.embeddings[X_test[i][0]]).reshape(-1, 1)
            end_node_emb = np.array(
                self.embeddings[X_test[i][1]]).reshape(-1, 1)
            # ranging from [-1, +1]
            score = cosine_similarity(start_node_emb, end_node_emb)
            # switch to prob... however, we may also directly y_score = score
            Y_probs.append((score + 1) / 2.0)
            # in sklearn roc... which yields the same reasult
        roc = roc_auc_score(y_true=Y_true, y_score=Y_probs)
        if roc < 0.5:
            roc = 1.0 - roc  # since lp is binary clf task, just predict the opposite if<0.5
        print("roc=", "{:.9f}".format(roc))


def cosine_similarity(a, b):
    from numpy import dot
    from numpy.linalg import norm
    ''' cosine similarity; can be used as score function; vector by vector; 
        If consider similarity for all pairs,
        pairwise_similarity() implementation may be more efficient
    '''
    a = np.reshape(a, -1)
    b = np.reshape(b, -1)
    if norm(a) * norm(b) == 0:
        return 0.0
    else:
        return dot(a, b) / (norm(a) * norm(b))


def read_node_label_downstream(filename):
    """ may be used in node classification task;
        part of labels for training clf and
        the result served as ground truth;
        note: similar method can be found in graph.py -> read_node_label
    """
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        line = fin.readline()
        if line == '':
            break
        vec = line.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def main(embeddings):
    # if embeddings is not None:
    #     embedding_path = '../medium_result/node_embedding_90.npy'
    #     embeddings = np.load(embedding_path)
    np.random.seed(random.randint(0, 10000))
    train_x, train_y = [], []
    for line in open('../data/label_train.csv').readlines():
        elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
        train_x.append(embeddings[elements[0]])
        train_y.append(elements[1])
    test_x, test_y = [], []
    for line in open('../data/label_test.csv').readlines():
        elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
        test_x.append(embeddings[elements[0]])
        test_y.append(elements[1])

    X, Y = read_node_label_downstream('../data/mit_label.txt')
    X = list(map(lambda item: int(item), X))
    ds_task = ncClassifier(vectors=embeddings,
                           clf=LogisticRegression())  # use Logistic Regression as clf; we may choose SVM or more advanced ones
    return ds_task.split_train_evaluate(X, Y, 0.5)


############################################################################

# classifier = SVC(C=1.0, kernel='rbf')
# classifier = MLPClassifier(hidden_layer_sizes=(512), max_iter=15)
# classifier.fit(train_x, train_y)
# score = classifier.score(train_x, train_y)
# print(score)
#
# # score = classifier.score(test_x, test_y)
# print(score)


if __name__ == '__main__':
    # walks = json.load(open('../data/node2vec_walk_path.json'))
    # walks = [[str(i) for i in walk] for walk in walks]
    # model = Word2Vec(walks, size=128, window=10, min_count=0, sg=1, workers=60,
    #                  iter=10)
    # model.save('../data/gensim_n2c.emb')

    w2c_model = Word2Vec.load('../data/gensim_n2c.emb')
    model_embedding = np.zeros((5298, 128))
    for i in range(5298):
        if str(i) in w2c_model.wv.index2word:
            model_embedding[i] = w2c_model[str(i)]
    n2c = []
    for _ in range(100):
        n2c.append(main(model_embedding[:, :128])['micro'])
    print(np.mean(n2c))

    # main(np.load('../medium_result/node_embedding_attr.npy'))
