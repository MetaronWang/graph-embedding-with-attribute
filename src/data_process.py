import json

import numpy as np


def generate_adjacent_matrix():
    lines = open('../data/adjlist.csv').readlines()
    node_num = len(lines)
    matrix = np.zeros((node_num, node_num), dtype=np.int)
    for line in lines:
        data = line.replace("\n", "").split(",")
        if len(data) <= 1 or data[-1] == '':
            print(line)
            continue
        elements = list(map(lambda item: int(item), data))
        for i in elements[1:]:
            matrix[elements[0], i] = 1
            matrix[i, elements[0]] = 1
    np.save('../data/adjacent_matrix.npy', matrix)


def generate_attributes():
    lines = open('../data/attr.csv').readlines()
    node_num = len(lines)
    attr_num = len(lines[0].split(',')) - 1
    attributes = np.zeros((node_num, attr_num), dtype=np.int)
    for line in lines:
        data = line.replace("\n", "").split(",")
        if len(data) <= 1 or data[-1] == '':
            print(line)
            continue
        elements = list(map(lambda item: int(item), data))
        for i in range(len(elements) - 1):
            attributes[elements[0], i] = elements[i + 1]
    np.save('../data/attributes.npy', attributes)


def generate_edge_list():
    lines = open('../data/adjlist.csv').readlines()
    edges = set()
    for line in lines:
        data = line.replace("\n", "").split(",")
        if len(data) <= 1 or data[-1] == '':
            print(line)
            continue
        elements = list(map(lambda item: int(item), data))
        for i in elements[1:]:
            head, tail = min(elements[0], i), max(elements[0], i)
            edges.add((head, tail))
    out = ''
    for i in list(edges):
        out += '{} {}\n'.format(i[0], i[1])
    open('../data/adjacent.edgelist', 'w', encoding='utf-8').write(out)


def generate_attr_vocabulary():
    lines = open('../data/attr.csv').readlines()
    attr_num = len(lines[0].split(',')) - 1
    attributes = [set() for _ in range(attr_num)]
    for line in lines:
        data = line.replace("\n", "").split(",")
        if len(data) <= 1 or data[-1] == '':
            print(line)
            continue
        elements = list(map(lambda item: int(item), data))
        for i in range(len(elements) - 1):
            attributes[i].add(elements[i+1])
    for i in range(attr_num):
        values = list(attributes[i])
        dict = {}
        reverse_dict = {}
        for key in range(len(values)):
            dict[values[key]] = key
            reverse_dict[key] = values[key]
        open('../data/attr_{}_vocab.json'.format(i), 'w').write(json.dumps(dict, indent=2))
        open('../data/attr_{}_reverse_vocab.json'.format(i), 'w').write(json.dumps(reverse_dict, indent=2))



if __name__ == '__main__':
    # generate_edge_list()
    generate_attr_vocabulary()