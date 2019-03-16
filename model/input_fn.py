"""
Here we define inputs to the model
"""

import tensorflow as tf
import pandas as pd


def build_vocab(file_name):
    tokens = tf.contrib.lookup.index_table_from_file(
        file_name,
        num_oov_buckets=0,
        default_value=1,
        delimiter='\n',
        name='vocab'
    )

    return tokens


def vectorize(string, vocab, seq_len):
    splitted = tf.string_split([string]).values
    vectorized = vocab.lookup(splitted)
    vectorized = vectorized[:seq_len]
    return vectorized


def input_fn(data_path, params, train_time=True):

    data = pd.read_csv(data_path)
    dataset = tf.data.Dataset.from_tensor_slices(({'tokens': data['text'].values}, data['labels'].values))

    vocab = build_vocab(params['vocab_path'])
    pad_value = vocab.lookup(tf.constant('<PAD>'))
    fake_padding = tf.constant(-1, dtype=tf.int64)

    if train_time:
        dataset = dataset.shuffle(params['train_size'])
        dataset = dataset.repeat(params['num_epochs'])

    dataset = dataset.map(lambda feats, labs: (vectorize(feats['tokens'], vocab, params['seq_len']), labs))

    padded_shapes = (tf.TensorShape([params['seq_len']]), tf.TensorShape([]))
    padding_values = (pad_value, fake_padding)

    dataset = dataset.padded_batch(
        batch_size=params['batch_size'],
        padded_shapes=padded_shapes,
        padding_values=padding_values
    )

    dataset = dataset.map(lambda x, y: ({'x': x, 'y': y}, y))
    dataset = dataset.prefetch(buffer_size=2)

    return dataset
