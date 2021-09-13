import json
import random
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize
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

    def train_evaluate(self, Y):
        train_x, train_y = [], []
        for line in open('../data/label_train.csv').readlines():
            elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
            train_x.append(elements[0])
            train_y.append([str(elements[1])])
        test_x, test_y = [], []
        for line in open('../data/label_test.csv').readlines():
            elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
            test_x.append(elements[0])
            test_y.append([str(elements[1])])
        self.train(train_x, train_y, Y)
        return self.evaluate(test_x, test_y)

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
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        print('Logistic Regression & ${}$ & ${}$ \\\\'.format(results["micro"] * 10000 // 1 / 100,
                                                              results["macro"] * 10000 // 1 / 100))


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


class Classifier:
    def __init__(self, embeddings, train_rate=None):
        if train_rate:
            x, y = [], []
            for line in open('../data/label_train.csv').readlines():
                elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
                x.append(embeddings[elements[0]])
                y.append(elements[1])
            for line in open('../data/label_test.csv').readlines():
                elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
                x.append(embeddings[elements[0]])
                y.append(elements[1])
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, test_size=1-train_rate,
                                                                                    random_state=1088)
        else:
            self.train_x, self.train_y = [], []
            for line in open('../data/label_train.csv').readlines():
                elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
                self.train_x.append(embeddings[elements[0]])
                self.train_y.append(elements[1])
            self.test_x, self.test_y = [], []
            for line in open('../data/label_test.csv').readlines():
                elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
                self.test_x.append(embeddings[elements[0]])
                self.test_y.append(elements[1])

    def svm_train_eval(self):
        cls = SVC()
        cls.fit(self.train_x, self.train_y)
        predict_y = cls.predict(self.test_x)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(self.test_y, predict_y, average=average)
        print('SVM & ${}$ & ${}$ \\\\'.format(results["micro"] * 10000 // 1 / 100, results["macro"] * 10000 // 1 / 100))

    def dt_train_eval(self):
        cls = DecisionTreeClassifier(max_depth=15)
        cls.fit(self.train_x, self.train_y)
        predict_y = cls.predict(self.test_x)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(self.test_y, predict_y, average=average)
        print('Decision Tree & ${}$ & ${}$ \\\\'.format(results["micro"] * 10000 // 1 / 100,
                                                        results["macro"] * 10000 // 1 / 100))

    def rf_train_eval(self):
        cls = RandomForestClassifier()
        cls.fit(self.train_x, self.train_y)
        predict_y = cls.predict(self.test_x)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(self.test_y, predict_y, average=average)
        print('Random Forest & ${}$ & ${}$ \\\\'.format(results["micro"] * 10000 // 1 / 100,
                                                        results["macro"] * 10000 // 1 / 100))

    def nb_train_eval(self):
        cls = BernoulliNB()
        cls.fit(self.train_x, self.train_y)
        predict_y = cls.predict(self.test_x)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = f1_score(self.test_y, predict_y, average=average)
        print('Naive Bayes & ${}$ & ${}$ \\\\'.format(results["micro"] * 10000 // 1 / 100,
                                                      results["macro"] * 10000 // 1 / 100))


def get_embedding():
    node_emb = Word2Vec.load('../data/gensim_n2c.emb')
    attrs_emb = [Word2Vec.load('../data/attr_{}_n2c.emb'.format(index)) for index in range(6)]
    base_dimension = 3
    node2vec_dimension = 80
    shapes = [[6, base_dimension], [3, base_dimension], [43, base_dimension * 2], [44, base_dimension * 2],
              [64, base_dimension * 2], [2506, base_dimension * 8]]
    model_embedding = np.zeros((5298, node2vec_dimension + base_dimension * 16))
    origin_attributes_list = np.load('../data/attributes.npy')
    vocab = [json.load(open('../data/attr_{}_vocab.json'.format(index))) for index in range(len(shapes))]
    for n in range(len(origin_attributes_list)):
        for a in range(len(shapes)):
            origin_attributes_list[n, a] = vocab[a][str(origin_attributes_list[n, a])]
    for i in range(5298):
        if str(i) in node_emb.wv.index2word:
            model_embedding[i, 0:node2vec_dimension] = node_emb[str(i)]
            current_index = node2vec_dimension
            for attr_i in range(6):
                try:
                    model_embedding[i, current_index:current_index + shapes[attr_i][1]] = attrs_emb[attr_i][
                        str(origin_attributes_list[i, attr_i])]
                except KeyError:
                    model_embedding[i, current_index:current_index + shapes[attr_i][1]] = np.zeros(shapes[attr_i][1])
                    print("SS")
                current_index += shapes[attr_i][1]
    return model_embedding


if __name__ == '__main__':
    embedding = get_embedding()#[:, :80]
    X, Y = read_node_label_downstream('../data/mit_label.txt')
    ncClassifier(vectors=embedding, clf=LogisticRegression()).train_evaluate(Y)
    clf = Classifier(embedding)
    clf.svm_train_eval()
    clf.nb_train_eval()
    clf.dt_train_eval()
    clf.rf_train_eval()
