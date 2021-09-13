import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def main(matrix):
    G = nx.from_numpy_matrix(matrix)
    pos = nx.spring_layout(G, iterations=10, threshold=10e-2)
    plt.figure(figsize=(10,10), dpi=500)  # 设置画布大小
    nx.draw_networkx_nodes(G, pos, node_size=1)  # 画节点
    nx.draw_networkx_edges(G, pos, width=0.1)  # 画边
    plt.show()


if __name__ == '__main__':
    # main(np.load('../data/adjacent_matrix.npy'))
    attributes = np.load('../data/attributes.npy')
    attributes = np.transpose(attributes)
    for i in range(len(attributes)):
        print(set(attributes[i]))
        print(len(set(attributes[i])))