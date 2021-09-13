from typing import List

import tensorflow as tf
import numpy as np
import json
import random
import math
import os
import collections

from tensorflow.python.ops import array_ops, nn_ops, candidate_sampling_ops
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tqdm import tqdm
from multiprocessing import Process, Manager, Pool

import sklearn_classifier


def generate_path_skip_gram(paths, num_skips, skip_window, path_index, return_dict):
    # num_skips = 1
    assert num_skips <= 2 * skip_window
    span: int = 2 * skip_window + 1  # [ skip_window target skip_window ]
    length: int = 0
    for path in paths:
        length += (len(path) - span + 1) * num_skips
    train_data = np.ndarray(shape=(length,), dtype=np.int32)
    train_label = np.ndarray(shape=(length, 1), dtype=np.int32)
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    train_index = 0
    for path in paths:
        data_index = 0
        buffer.extend(path[data_index:data_index + span])
        data_index += span
        while True:
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                train_data[train_index * num_skips + j] = buffer[skip_window]
                train_label[train_index * num_skips + j] = buffer[
                    context_word]  # np.array([buffer[context_word] for context_word in context_words])

            train_index += 1
            if data_index == len(path):
                break
            else:
                buffer.append(path[data_index])
                data_index += 1
    return_dict[path_index] = (train_data, train_label)


def generate_skip_gram_epoch(paths: List[List[int]], num_skips: int, skip_window: int, return_dict):
    assert num_skips <= 2 * skip_window
    manager = Manager()
    result_dict = manager.dict()
    pool = Pool(processes=60)
    path_term = np.array_split(paths, 60)
    for index in range(len(path_term)):
        pool.apply_async(generate_path_skip_gram, args=(path_term[index], num_skips, skip_window, index, result_dict))
    pool.close()
    pool.join()
    train_data = np.concatenate([result_dict[index][0] for index in range(len(path_term))])
    train_label = np.concatenate([result_dict[index][1] for index in range(len(path_term))])
    return_dict['data'] = train_data
    return_dict['label'] = train_label


def generate_node_vocab(paths):
    node_num = [0 for _ in range(5298)]
    for path in paths:
        for node in path:
            node_num[node] += 1
    node_rank = list(range(5298))
    node_rank.sort(key=lambda item: node_num[item], reverse=True)
    node_vocab, node_vocab_reverse = {}, {}
    for rank, node in enumerate(node_rank):
        node_vocab[node] = rank
        node_vocab_reverse[rank] = node
    new_paths = [[node_vocab[node] for node in path] for path in paths]
    return node_vocab, node_vocab_reverse, new_paths


class AttributeNode2Vec:
    def __init__(self, attributes_shapes, attributes_list, node_shape, skip_window, batch_size, epoch_num,
                 number_sampled, use_attr=True):
        self.node_matrix_shape = node_shape
        self.attribute_matrices_shape = attributes_shapes
        self.batch_size = batch_size
        self.skip_windows = skip_window
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
            self.node_input = tf.placeholder(tf.int32, [None], name='node_input')
            self.node_label = tf.placeholder(tf.int64, [None, 1], name='label_input')
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
            self.embedding = tf.nn.embedding_lookup(self.all_embedding, self.node_input)
            self.hidden_shape = [512, 256]
            self.hidden_kernel_matrices = [
                tf.get_variable("hidden_kernel_{}".format(i),
                                [self.embedding_dim if i == 0 else self.hidden_shape[i - 1], self.hidden_shape[i]]) for
                i in range(len(self.hidden_shape))
            ]
            self.hidden_bias_matrices = [
                tf.get_variable("hidden_bias_{}".format(i), [self.hidden_shape[i]]) for i in
                range(len(self.hidden_shape))
            ]
            # for i in range(len(self.hidden_kernel_matrices)):
            #     self.embedding = tf.matmul(self.embedding, self.hidden_kernel_matrices[i]) + \
            #                      self.hidden_bias_matrices[i]
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal([self.node_matrix_shape[0], self.hidden_shape[-1]],
                                    stddev=1.0 / math.sqrt(self.hidden_shape[-1])))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([self.node_matrix_shape[0]]))
        # with tf.name_scope('loss'):
        #
        #     logits, labels = _compute_sampled_logits(
        #         weights=nce_weights,
        #         biases=nce_biases,
        #         labels=self.node_label,
        #         inputs=self.embedding,
        #         num_sampled=self.number_sampled,
        #         num_classes=self.node_matrix_shape[0],
        #         num_true=2 * self.skip_windows,
        #         subtract_log_q=True,
        #         remove_accidental_hits=True,
        #         partition_strategy="mod",
        #         name="sampled_softmax_loss"
        #     )
        #     # labels = array_ops.stop_gradient(labels, name="labels_stop_gradient")
        #     sampled_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        #         labels=labels, logits=logits)
        #     self.loss = tf.reduce_mean(sampled_losses)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=self.node_label,
                    inputs=self.embedding,
                    num_sampled=self.number_sampled,
                    num_true=1,  # 2 * self.skip_windows,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    paths = json.load(open('../data/node2vec_walk_path.json'))
    # paths = json.load(open('../data/deepwalk_walk_path.json'))
    node_vocab, node_vocab_reverse, paths = generate_node_vocab(paths)
    attributes_list = np.array([origin_attributes_list[node_vocab_reverse[rank]] for rank in range(len(node_vocab))])
    # AttributeNode2Vec(shapes, attributes_list, [5298, 128], 5, 256, 100, 128, use_attr=False)
    length: int = 0
    for path in paths:
        length += (len(path) - 2 * 3) * 4
    print(length)
    manager = Manager()
    return_dict = manager.dict()
    windows = 5
    skip_num = 2
    p = Process(target=generate_skip_gram_epoch, args=(paths, skip_num, windows, return_dict))
    p.start()
    max_alpha, min_alpha = 0.0015, 0.0005
    max_batch, min_batch = 256, 64
    epoch_num = 1000
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=tf.Graph()) as sess:
        an2v = AttributeNode2Vec(shapes, attributes_list, [5298, 128], windows, 256, 100, 1024, use_attr=True)
        sess.run(tf.global_variables_initializer())

        for epoch_index in range(epoch_num):
            p.join()
            learning_rate = 0.001# max_alpha - epoch_index * (max_alpha - min_alpha) / epoch_num
            train_datas, train_labels = return_dict['data'], return_dict['label']
            batch_num = 64  # max_batch - int(epoch_index * (max_batch - min_batch) / epoch_num)
            p = Process(target=generate_skip_gram_epoch, args=(paths, skip_num, windows, return_dict))
            p.start()
            data_batches = np.array_split(train_datas, batch_num)
            label_batches = np.array_split(train_labels, batch_num)
            epoch_loss = []
            for index in tqdm(range(len(data_batches))):
                data_batch, label_batch = data_batches[index], label_batches[index]
                feed_dict = {an2v.node_input.name: data_batch, an2v.node_label.name: label_batch,
                             an2v.learning_rate.name: learning_rate}
                ret = sess.run([an2v.loss, an2v.update], feed_dict=feed_dict)
                epoch_loss.append(ret[0])
                # if index % 64 == 0:
                #     rank_embedding = sess.run([an2v.normalized_embeddings])[0]
                #     final_embedding = np.array([rank_embedding[node_vocab[node]] for node in range(5298)])
                #     sklearn_classifier.main(final_embedding)
            print(epoch_index, np.mean(epoch_loss))
            if (epoch_index + 1) % 1 == 0:
                rank_embedding = sess.run([an2v.normalized_embeddings])[0]
                final_embedding = np.array([rank_embedding[node_vocab[node]] for node in range(5298)])
                print(sklearn_classifier.main(final_embedding))
                # np.save('../medium_result/node_embedding.npy', final_embedding)
            # np.save('../medium_result/node_embedding_{}.npy'.format(epoch_index + 1), final_embedding)

        # p.join()


if __name__ == '__main__':
    main()
