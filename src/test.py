import json

import numpy as np

if __name__ == '__main__':
    train_y = []
    for line in open('../data/label_train.csv').readlines():
        elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
        train_y.append(elements[1])
    set_1 = set(train_y)
    train_y = list(set_1)
    dict_y = {}
    dict_y_reverse = {}
    for v, k in enumerate(train_y):
        dict_y[k] = v
        dict_y_reverse[v] = k
    open('../data/label_vocab.json', 'w').write(json.dumps(dict_y, indent=2))
    open('../data/label_vocab_reverse.json', 'w').write(json.dumps(dict_y_reverse, indent=2))

    for line in open('../data/label_test.csv').readlines():
        elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
        train_y.append(elements[1])
    set_2 = set(train_y)
    print(len(set(train_y)))
    set3 = set_2 - set_1
    for i in train_y:
        if i in set3:
            print(i)
