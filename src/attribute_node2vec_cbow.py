from typing import List

import tensorflow as tf
import numpy as np
import json
import random
import math
import os
import collections
from tqdm import tqdm
from multiprocessing import Process, Manager, Pool

import sklearn_classifier


def generate_path_cbow(paths, bag_window, path_index, return_dict):
    span: int = 2 * bag_window + 1  # [ skip_window target skip_window ]
    length: int = 0
    for path in paths:
        length += (len(path) - span + 1)
    train_data = np.ndarray(shape=(length, span - 1), dtype=np.int32)
    train_label = np.ndarray(shape=(length, 1), dtype=np.int32)
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    train_index = 0
    for path in paths:
        data_index = 0
        buffer.extend(path[data_index:data_index + span])
        data_index += span
        while True:
            context_words = [w for w in range(span) if w != bag_window]
            train_data[train_index] = [buffer[context_word] for context_word in context_words]
            train_label[train_index, 0] = buffer[bag_window]
            train_index += 1
            if data_index == len(path):
                break
            else:
                buffer.append(path[data_index])
                data_index += 1
    return_dict[path_index] = (train_data, train_label)


def generate_cbow_epoch(paths: List[List[int]], bag_window: int):
    manager = Manager()
    result_dict = manager.dict()
    pool = Pool(processes=60)
    path_term = np.array_split(paths, 60)
    for index in range(len(path_term)):
        pool.apply_async(generate_path_cbow, args=(path_term[index], bag_window, index, result_dict))
    pool.close()
    pool.join()
    train_data = np.concatenate([result_dict[index][0] for index in range(len(path_term))])
    train_label = np.concatenate([result_dict[index][1] for index in range(len(path_term))])
    return train_data, train_label


def generate_node_vocab(paths):
    node_num = [0 for _ in range(5298)]
    for path in paths:
        for node in path:
            node_num[node] += 1
    node_rank = list(range(5298))
    node_rank.sort(key=lambda item: node_num[item], reverse=True)
    # node_rank = [node for node in node_rank if node_num[node] > 0]
    node_vocab, node_vocab_reverse = {}, {}
    for rank, node in enumerate(node_rank):
        node_vocab[node] = rank
        node_vocab_reverse[rank] = node
    new_paths = [[node_vocab[node] for node in path] for path in paths]
    return node_vocab, node_vocab_reverse, new_paths


class AttributeNode2Vec:
    def __init__(self, attributes_shapes, attributes_list, node_shape, bag_window, batch_size, epoch_num,
                 number_sampled, use_attr=True):
        self.node_matrix_shape = node_shape
        self.attribute_matrices_shape = attributes_shapes
        self.batch_size = batch_size
        self.bag_windows = bag_window
        self.epoch_num = epoch_num
        self.number_sampled = number_sampled
        self.use_attr = use_attr
        self.all_embedding, normalized_embeddings, self.node_embedding, self.attribute_embeddings = None, None, None, [
            None for _ in
            range(len(attributes_shapes))]
        self.node_label, self.node_input = None, None
        self.loss, self.update = None, None
        self.embedding_dim = node_shape[1]
        if self.use_attr:
            for shape in attributes_shapes:
                self.embedding_dim += shape[1]
        self.attributes_list = np.transpose(attributes_list)
        self.init_calculate_graph()
        self.initial_valid()

    def init_calculate_graph(self):
        with tf.name_scope('inputs'):
            self.node_input = tf.placeholder(tf.int32, [None, 2 * self.bag_windows], name='node_input')
            self.node_label = tf.placeholder(tf.int32, [None, 1], name='label_input')
            self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        with tf.name_scope('embeddings'):
            self.node_embedding = tf.get_variable("node_embedding", self.node_matrix_shape)
            for i in range(len(self.attribute_matrices_shape)):
                self.attribute_embeddings[i] = tf.get_variable("attribute_embedding_{}".format(i),
                                                               self.attribute_matrices_shape[i])
            self.all_embedding = self.node_embedding
            if self.use_attr:
                for i in range(len(self.attribute_matrices_shape)):
                    temp = tf.gather(self.attribute_embeddings[i], self.attributes_list[i])
                    self.all_embedding = tf.concat((self.all_embedding, temp), axis=1)
            # self.embedding = tf.nn.embedding_lookup(self.all_embedding, self.node_input)

        embeds = tf.nn.embedding_lookup(self.all_embedding, self.node_input)
        hidden_weights_1 = tf.get_variable("hidden_embedding_1", [self.bag_windows * 2, 32])
        # hidden_weights_2 = tf.get_variable("hidden_embedding_2", [64, 256])
        # bag_weights = tf.get_variable("bag_embedding", [32])
        # embeds = tf.einsum('bij,ik->bkj', embeds, hidden_weights_1)
        # embeds = tf.einsum('bij,ik->bkj', embeds, hidden_weights_2)
        # avg_embed = tf.einsum('bij,i->bj', embeds, bag_weights)
        avg_embed = tf.reduce_mean(embeds, 1, keep_dims=False)
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal([self.node_matrix_shape[0], self.embedding_dim],
                                    stddev=1.0 / math.sqrt(self.embedding_dim)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([self.node_matrix_shape[0]]))

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=self.node_label,
                    inputs=avg_embed,
                    num_sampled=self.number_sampled,
                    num_classes=self.node_matrix_shape[0]))
        # tf.summary.scalar('loss', self.loss)
        with tf.name_scope('optimizer'):
            params = tf.trainable_variables()
            # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            gradients = tf.gradients(self.loss, params)
            self.update = opt.apply_gradients(zip(gradients, params))

        norm = tf.sqrt(tf.reduce_sum(tf.square(self.all_embedding), 1, keepdims=True))
        self.normalized_embeddings = self.all_embedding / norm

    def initial_valid(self):
        pass


def main():
    base_dimension = 8
    shapes = [[6, base_dimension], [3, base_dimension], [43, base_dimension * 2], [44, base_dimension * 2],
              [64, base_dimension * 2], [2506, base_dimension * 8]]
    origin_attributes_list = np.load('../data/attributes.npy')
    vocab = [json.load(open('../data/attr_{}_vocab.json'.format(index))) for index in range(len(shapes))]
    for n in range(len(origin_attributes_list)):
        for a in range(len(shapes)):
            origin_attributes_list[n, a] = vocab[a][str(origin_attributes_list[n, a])]
    gpu_options = tf.GPUOptions(allow_growth=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # paths = json.load(open('../data/deepwalk_walk_path.json'))
    paths = json.load(open('../data/node2vec_walk_path.json'))
    node_vocab, node_vocab_reverse, paths = generate_node_vocab(paths)
    attributes_list = np.array([origin_attributes_list[node_vocab_reverse[rank]] for rank in range(len(node_vocab))])
    length: int = 0
    for path in paths:
        length += (len(path) - 2 * 3) * 4
    print(length)
    windows = 5
    max_alpha, min_alpha = 0.025, 0.0001
    max_batch, min_batch = 128, 64
    epoch_num = 100
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=tf.Graph()) as sess:
        an2v = AttributeNode2Vec(shapes, attributes_list, [len(node_vocab), 128], windows, 256, 100, 128,
                                 use_attr=False)
        sess.run(tf.global_variables_initializer())
        train_datas, train_labels = generate_cbow_epoch(paths, windows)
        for epoch_index in range(epoch_num):
            learning_rate = 0.001 # max_alpha - epoch_index * (max_alpha - min_alpha) / epoch_num
            batch_num = 384  # max_batch - int(epoch_index * (max_batch - min_batch) / epoch_num)
            data_batches = np.array_split(train_datas, batch_num)
            label_batches = np.array_split(train_labels, batch_num)
            epoch_loss = []
            for index in tqdm(range(len(data_batches))):
                data_batch, label_batch = data_batches[index], label_batches[index]
                feed_dict = {an2v.node_input.name: data_batch, an2v.node_label.name: label_batch,
                             an2v.learning_rate.name: learning_rate}
                ret = sess.run([an2v.loss, an2v.update], feed_dict=feed_dict)
                epoch_loss.append(ret[0])
            print(epoch_index, np.mean(epoch_loss))
            if (epoch_index + 1) % 1 == 0:
                rank_embedding = sess.run([an2v.normalized_embeddings])[0]
                # final_embedding = np.zeros((5298, 256))
                # for i in range(len(node_vocab)):
                #     final_embedding[node_vocab_reverse[i]] = rank_embedding[i]
                final_embedding = np.array([rank_embedding[node_vocab[node]] for node in range(5298)])
                sklearn_classifier.main(final_embedding)
                # np.save('../medium_result/node_embedding.npy', final_embedding)
            # np.save('../medium_result/node_embedding_{}.npy'.format(epoch_index + 1), final_embedding)

        # p.join()


if __name__ == '__main__':
    main()
