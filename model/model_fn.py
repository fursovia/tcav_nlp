"""
Here we define architecture, loss, metrics and so on
"""

import sys
sys.path.append('..')
import tensorflow as tf
import os
from model.utils import get_yaml_config, vectorize, one_hot
import numpy as np


def build_model(emb_matrix, features, params):

    embeddings = tf.nn.embedding_lookup(emb_matrix, features['x'], name='emb_matrix_lookup')

    out = tf.reduce_max(embeddings, axis=1, name='dim_reduction')
    out = tf.layers.dense(out, 256, name='bottleneck')
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, params['num_classes'], name='output_logits')

    return out


def model_fn(features, labels, mode, params):
    # is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):

        emb_matrix = tf.get_variable('embedding_matrix',
                                     shape=[params['vocab_size'], params['emb_dim']],
                                     dtype=tf.float64)

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        logits = build_model(emb_matrix, features, params)
        preds = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits, 'preds': preds}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(labels, depth=params['num_classes'])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(logits, axis=1), labels)))

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.variable_scope('metrics'):

            eval_metric_ops = {'accuracy': tf.metrics.mean(accuracy)}

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    optimizer_fn = tf.train.AdamOptimizer(params['learning_rate'])

    global_step = tf.train.get_global_step()

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=params['learning_rate'],
        optimizer=optimizer_fn,
        name='optimizer')

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


class ModelWrapper:

    def __init__(self, model_path, endpoints):
        self.model_path = model_path
        self.endpoints = endpoints
        self.emb_matrix = None
        self.outputs = None
        self.output = None  # logits
        self.loss = None
        self.bottleneck = None
        self.bottleneck_grad = None

        ckpt_file = open(os.path.join(self.model_path, "./checkpoint"), "r")
        lines = ckpt_file.readlines()
        ckpt_file.close()
        self.path_to_ckpt = os.path.join(self.model_path, lines[-1].split()[-1][1:-1])

        self.config = get_yaml_config(os.path.join(self.model_path, 'config.yaml'))

        with open(self.config['vocab_path'], 'r') as file:
            self.vocab = dict()
            c = 0
            for line in file:
                self.vocab[line.strip()] = c
                c += 1

        self.import_graph()
        self.get_tensors()

    def import_graph(self):
        tf.reset_default_graph()
        graph = tf.get_default_graph()

        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as ss:
            saver = tf.train.import_meta_graph(self.path_to_ckpt + '.meta')
            saver.restore(ss, self.path_to_ckpt)

            self.emb_matrix = ss.run(graph.get_tensor_by_name('embeddings/embedding_matrix:0'))
            self.outputs = {key: graph.get_operation_by_name(val).outputs[0] for key, val in self.endpoints.items()}

    def get_tensors(self):
        graph = tf.get_default_graph()
        self.bottleneck = self.outputs['bottleneck']
        self.output = self.outputs['output']
        self.loss = graph.get_operation_by_name('softmax_cross_entropy_loss/Cast_1').outputs[0]
        self.bottleneck_grad = tf.gradients(self.loss, self.bottleneck)[0]

    def get_input(self, text):
        examples = vectorize(text, self.config['seq_len'], self.vocab, self.emb_matrix)
        examples = examples.reshape(1, self.config['seq_len'], self.config['emb_dim'])
        inp = examples.max(axis=1)
        return inp

    def calculate_grad(self, sess, text, labels):
        labels = labels.reshape(-1, 1)
        onehot = one_hot(labels, self.config['num_classes'])
        inp = self.get_input(text)

        sess.run(tf.global_variables_initializer())
        grads = sess.run(self.bottleneck_grad, {'one_hot:0': onehot, self.outputs['input_'].name: inp})

        return grads

    def calculate_output(self, sess, text):
        inp = self.get_input(text)

        sess.run(tf.global_variables_initializer())

        logits = sess.run(self.output, {self.outputs['input_'].name: inp})

        return logits
