import numpy as np
import networkx as nx
import random
import json
from multiprocessing import Pool, Manager, Process
from gensim.models import Word2Vec
from sklearn_classifier import main


class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                               alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def read_graph(weighted=False, input_file='../data/adjacent.edgelist', directed=False):
    """
    Reads the input network in networkx.
    """
    if weighted:
        G = nx.read_edgelist(input_file, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input_file, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    if not directed:
        G = G.to_undirected()
    return G


def node_walk():
    nx_G = read_graph(input_file='../data/adjacent.edgelist')
    G = Graph(nx_G, is_directed=False, p=0.5, q=0.5)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=24, walk_length=120)
    open('../data/node2vec_walk_path.json', 'w', encoding='utf-8').write(json.dumps(walks, indent=2))


def one_attr_walk(index):
    edge_num = len(open('../data/attr_{}.edgelist'.format(index)).readlines())
    nx_G = read_graph(input_file='../data/attr_{}.edgelist'.format(index), weighted=False)
    G = Graph(nx_G, is_directed=False, p=0.5, q=0.5)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=24, walk_length=120)
    open('../data/attr_{}_2vec_walk_path.json'.format(index), 'w', encoding='utf-8').write(json.dumps(walks, indent=2))


def attr_walk():
    pool = Pool(processes=12)
    for i in range(6):
        pool.apply_async(one_attr_walk, args=(i,))
    pool.close()
    pool.join()


def attr2vec():
    base_dimension = 3
    shapes = [[6, base_dimension], [3, base_dimension], [43, base_dimension * 2], [44, base_dimension * 2],
              [64, base_dimension * 2], [2506, base_dimension * 8]]
    windows = [1, 1, 3, 3, 5, 5]
    for index in range(6):
        print("ATTR{}".format(index))
        walks = json.load(open('../data/attr_{}_2vec_walk_path.json'.format(index)))
        walks = [[str(i) for i in walk] for walk in walks]
        model = Word2Vec(walks, size=shapes[index][1], window=windows[index], min_count=0, sg=1, workers=72,
                         iter=50)
        model.save('../data/attr_{}_n2c.emb'.format(index))


def node2vec():
    walks = json.load(open('../data/node2vec_walk_path.json'))
    walks = [[str(i) for i in walk] for walk in walks]
    model = Word2Vec(walks, size=80, window=5, min_count=0, sg=1, workers=72,
                     iter=50)
    model.save('../data/gensim_n2c.emb')


def aggregate_embedding():
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
    manage = Manager()
    n2c, a2c, a = [], [], []
    for _ in range(10):
        a2c.append(main(model_embedding)['micro'])
        n2c.append(main(model_embedding[:, :80])['micro'])
        a.append(main(model_embedding[:, 80:])['micro'])
    print(np.mean(n2c), np.max(n2c))
    print(np.mean(a2c), np.max(a2c))
    print(np.mean(a), np.max(a))

    np.save('../medium_result/total_embedding.npy', model_embedding)
    np.save('../medium_result/node_embedding.npy', model_embedding[:, :80])
    np.save('../medium_result/attr_embedding.npy', model_embedding[:, 80:])


if __name__ == '__main__':
    p = Process(target=node_walk)
    p.start()
    attr_walk()
    p.join()
    print("Walk Finished")
    p = Process(target=node2vec)
    p.start()
    attr2vec()
    p.join()
    print("2vec Finished")
    aggregate_embedding()
