import random

import numpy as np
import json


def load_node_edge():
    edge_lists = []
    lines = open('../data/adjacent_full.edgelist').readlines()
    for line in lines:
        edge = line.replace('\n', '').split(' ')
        edge_lists.append((int(edge[0]), int(edge[1])))
    random.shuffle(edge_lists)
    return edge_lists[:len(edge_lists) * 10 // 10]


if __name__ == '__main__':
    base_dimension = 8
    attributes_list = np.load('../data/attributes.npy')
    shapes = [[6, base_dimension], [3, base_dimension], [43, base_dimension * 2], [44, base_dimension * 2],
              [64, base_dimension * 2], [2506, base_dimension * 8]]
    vocab = [json.load(open('../data/attr_{}_vocab.json'.format(index))) for index in range(len(shapes))]
    for n in range(len(attributes_list)):
        for a in range(len(shapes)):
            attributes_list[n, a] = vocab[a][str(attributes_list[n, a])]
    attributes_network = []
    node_edges = load_node_edge()
    the_node_edge_strings = ['{} {}'.format(edge[0], edge[1]) for edge in node_edges]
    open('../data/adjacent.edgelist', 'w', encoding='utf8').write('\n'.join(the_node_edge_strings))
    for index in range(len(shapes)):
        this_attr_edge_dict = {}
        for edge in node_edges:
            edge0, edge1 = min(edge), max(edge)
            string = "{} {}".format(attributes_list[edge0, index], attributes_list[edge1, index])
            if string in this_attr_edge_dict.keys():
                this_attr_edge_dict[string] += 1
            else:
                this_attr_edge_dict[string] = 1
        attributes_network.append(this_attr_edge_dict)
    for index in range(len(shapes)):
        this_attr_edge_dict = attributes_network[index]
        edgeStrings = ["{}".format(edge) for edge in this_attr_edge_dict]
        # edgeStrings = ["{} {}".format(edge, this_attr_edge_dict[edge]) for edge in this_attr_edge_dict]
        open('../data/attr_{}.edgelist'.format(index), 'w', encoding='utf8').write('\n'.join(edgeStrings))
