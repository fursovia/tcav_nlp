"""
Here we define architecture, loss, metrics and so on
"""

import tensorflow as tf


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

        emb_matrix = tf.get_variable('treatments_embeddings',
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
