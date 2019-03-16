"""
Starts the training process
"""

import os
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from model.model_fn import model_fn
from model.input_fn import input_fn
from model.utils import save_dict_to_yaml, get_yaml_config, calculate_metrics
from shutil import copyfile

parser = argparse.ArgumentParser()

parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-md', '--model_dir', default='experiments')
parser.add_argument('--seq_len', type=int, default=None)
parser.add_argument('--num_epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--learning_rate', type=float, default=None)


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(os.path.join(args.model_dir, 'config.yaml')):
        copyfile(os.path.join('experiments', 'config.yaml'), os.path.join(args.model_dir, 'config.yaml'))

    params = get_yaml_config(os.path.join(args.model_dir, 'config.yaml'))
    params['model_dir'] = args.model_dir
    params['data_dir'] = args.data_dir
    params['vocab_path'] = os.path.join(args.data_dir, 'vocab.txt')
    params['vocab_size'] = sum(1 for _ in open(params['vocab_path'], 'r')) + 1
    data = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    params['train_size'] = data.shape[0]
    params['num_classes'] = data['labels'].nunique()

    params['seq_len'] = args.seq_len if args.seq_len is not None else params['seq_len']
    params['num_epochs'] = args.num_epochs if args.num_epochs is not None else params['num_epochs']
    params['batch_size'] = args.batch_size if args.batch_size is not None else params['batch_size']
    params['learning_rate'] = args.learning_rate if args.learning_rate is not None else params['learning_rate']

    config = tf.estimator.RunConfig(tf_random_seed=24,
                                    save_checkpoints_steps=int(params['train_size'] / params['batch_size']),
                                    keep_checkpoint_max=None,
                                    model_dir=args.model_dir,
                                    save_summary_steps=20)

    estimator = tf.estimator.Estimator(model_fn,
                                       params=params,
                                       config=config)

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(os.path.join(args.data_dir, 'train.csv'), params, True),
            max_steps=int((params['train_size'] / params['batch_size']) * params['num_epochs']),
        ),

        eval_spec=tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(os.path.join(args.data_dir, 'eval.csv'), params, False),
            steps=None,
            start_delay_secs=0,
            throttle_secs=60
        )
    )

    # SAVE PREDICTIONS

    eval_preds = estimator.predict(lambda: input_fn(os.path.join(args.data_dir, 'eval.csv'), params, False))

    eval_logits = []
    probs = []
    for p in eval_preds:
        eval_logits.append(p['logits'])
        probs.append(p['preds'])

    eval_logits = np.array(eval_logits, np.float64).reshape(-1, 2)
    probs = np.array(probs, np.float64).reshape(-1, 2)

    np.save(os.path.join(args.model_dir, 'eval_logits.npy'), eval_logits)
    np.save(os.path.join(args.model_dir, 'eval_probs.npy'), probs)

    # CALCULATE METRICS

    labels = pd.read_csv(os.path.join(args.data_dir, 'eval.csv'))['target'].values
    metrics = calculate_metrics(probs, labels)

    for key, val in metrics.items():
        params[key] = str(val)
        print(f'{key} = {val}')

    save_dict_to_yaml(params, os.path.join(args.model_dir, 'config.yaml'))
