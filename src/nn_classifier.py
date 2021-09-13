import json
import os
import time

from sklearn import metrics
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class NN_Classifier:
    def __init__(self, node_embedding, attributes_shape, attributes_list, category_num, hidden_shape=None,
                 filter_shape=None, epoch_num=50, batch_size=128, learning_rate=1e-3, use_attr=False):
        if hidden_shape is None:
            hidden_shape = [512, 256]
        if filter_shape is None:
            filter_shape = [(32, 16), (16, 32), (16, 16), (16, 1)]
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)
            self.node_embedding = tf.constant(node_embedding, dtype=tf.float32)
            self.attribute_matrices_shape = attributes_shape
            self.category_num = category_num
            self.use_attr = use_attr
            self.attributes_list = np.transpose(attributes_list)
            self.node_shape = self.node_embedding.shape
            self.embedding_dim = self.node_shape[1]
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epoch_num = epoch_num
            if self.use_attr:
                for shape in self.attribute_matrices_shape:
                    self.embedding_dim += shape[1]
            self.hidden_shape = hidden_shape
            self.filter = filter_shape
            self.hidden_shape.append(self.category_num)
            self.init_calculate_graph()
            self.sess.run(tf.global_variables_initializer())

    def init_calculate_graph(self):
        with tf.name_scope('inputs'):
            self.node_input = tf.placeholder(tf.int32, [None], name='node_input')
            self.node_label = tf.placeholder(tf.int32, [None], name='node_label')
            self.attrs_input = []
            for i in range(len(self.attribute_matrices_shape)):
                self.attrs_input.append(tf.placeholder(tf.int32, [None], name='node_attr_{}'.format(i)))
        with tf.name_scope('embeddings'):
            self.attribute_embeddings = []
            for i in range(len(self.attribute_matrices_shape)):
                self.attribute_embeddings.append(
                    tf.get_variable("attribute_embedding_{}".format(i), self.attribute_matrices_shape[i]))
        self.all_embedding = self.node_embedding
        if self.use_attr:
            for i in range(len(self.attribute_matrices_shape)):
                temp = tf.gather(self.attribute_embeddings[i], self.attributes_list[i])
                self.all_embedding = tf.concat((self.all_embedding, temp), axis=1)
        self.embedding = tf.nn.embedding_lookup(self.all_embedding, self.node_input)

        self.hidden_kernel_matrices = [
            tf.get_variable("hidden_kernel_{}".format(i),
                            [self.embedding_dim if i == 0 else self.hidden_shape[i - 1], self.hidden_shape[i]]) for i in
            range(len(self.hidden_shape))
        ]
        self.hidden_bias_matrices = [
            tf.get_variable("hidden_bias_{}".format(i), [self.hidden_shape[i]]) for i in range(len(self.hidden_shape))
        ]
        self.filters_matrices = [
            tf.get_variable("filter_kernel_{}".format(i),
                            [self.filter[i][0], 1 if i == 0 else self.filter[i - 1][1], self.filter[i][1]]) for i in
            range(len(self.filter))]
        hidden_layer_out = tf.expand_dims(self.embedding, 2)
        with tf.name_scope('classifier'):
            for conv_filter in self.filters_matrices:
                hidden_layer_out = tf.nn.relu(
                    tf.nn.conv1d(hidden_layer_out, conv_filter, stride=1, padding='SAME'))

            hidden_layer_out = tf.squeeze(hidden_layer_out, 2)

            for i in range(len(self.hidden_kernel_matrices)):
                hidden_layer_out = tf.nn.relu(tf.matmul(hidden_layer_out, self.hidden_kernel_matrices[i]) + \
                                   self.hidden_bias_matrices[i])

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=hidden_layer_out, labels=self.node_label)
        )
        self.output = tf.nn.softmax(hidden_layer_out)
        with tf.name_scope('optimizer'):
            params = tf.trainable_variables()
            regularization_cost = 5e-3 * tf.reduce_sum([tf.nn.l2_loss(v) for v in params])
            self.l2_loss = self.loss + regularization_cost
            opt = tf.train.AdamOptimizer(self.learning_rate)
            gradients = tf.gradients(self.l2_loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, 10.0)
            self.update = opt.apply_gradients(zip(gradients, params))

    def train(self):
        train_x, train_y = [], []
        label_vocab = json.load(open('../data/label_vocab.json'))
        for line in open('../data/label_train.csv').readlines():
            elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
            train_x.append(elements[0])
            train_y.append(label_vocab[str(elements[1])])
        batches = [
            (train_x[i:i + self.batch_size], train_y[i:i + self.batch_size]) for i in
            range(0, len(train_y), self.batch_size)
        ]
        for epoch in range(self.epoch_num):
            losses = []
            for batch in batches:
                losses.append(self.sess.run([self.loss, self.update],
                                            feed_dict={
                                                self.node_input.name: batch[0],
                                                self.node_label.name: batch[1]
                                            })[0]
                              )
            print(epoch, np.mean(losses), self.valid()[1])
            # time.sleep(1)

    def valid(self):
        test_x, test_y = [], []
        label_vocab_reverse = json.load(open('../data/label_vocab_reverse.json'))
        for line in open('../data/label_test.csv').readlines():
            elements = list(map(lambda item: int(item), line.replace("\n", "").split(",")))
            test_x.append(elements[0])
            test_y.append(elements[1])
        result_probability = self.sess.run([self.output], feed_dict={self.node_input.name: test_x})[0]
        result = np.argmax(result_probability, axis=1)
        predict_y = [label_vocab_reverse[str(i)] for i in result]
        r = metrics.classification_report(test_y, predict_y)
        acc = metrics.accuracy_score(test_y, predict_y)
        averages = ["micro", "macro"]
        results = {}
        for average in averages:
            results[average] = metrics.f1_score(test_y, predict_y, average=average)
        print('MLP & ${}$ & ${}$ \\\\'.format(results["micro"] * 10000 // 1 / 100,
                                                        results["macro"] * 10000 // 1 / 100))
        return r, acc


def main():
    base_dimension = 8
    shapes = [[6, base_dimension], [3, base_dimension], [43, base_dimension * 2], [44, base_dimension * 2],
              [64, base_dimension * 2], [2506, base_dimension * 8]]
    attributes_list = np.load('../data/attributes.npy')
    node_embeddings = np.load('../medium_result/total_embedding.npy')
    category_num = len(json.load(open('../data/label_vocab.json')))
    nn_c = NN_Classifier(node_embeddings, shapes, attributes_list, category_num, epoch_num=20000, batch_size=128,
                         use_attr=False, hidden_shape=[2048], filter_shape=[])
    nn_c.train()
    # nn_c.valid()


if __name__ == '__main__':
    main()
