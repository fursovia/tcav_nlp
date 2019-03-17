import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import pandas as pd
import numpy as np
from nltk import ngrams
from collections import Counter
import pickle
from model.utils import clean
from model.model_fn import ModelWrapper
import tensorflow as tf
from cav import CAV


parser = argparse.ArgumentParser()

parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-md', '--model_dir', default='experiments')
parser.add_argument('-c', '--concept_names', default='путин россия сша')
parser.add_argument('-a', '--architecture', default='swem_max')
parser.add_argument('--ngrams', type=int, default=2)


swem_max_endpoints = dict(
    input_='model/dim_reduction',
    bottleneck='model/bottleneck/BiasAdd',
    probs='model/Softmax',
    output='model/output_logits/BiasAdd'
)


def get_ngrams(text, n=2):
    words = text.split()
    ngram_tokens = ngrams(words, n)
    return list(ngram_tokens)


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()

    if args.architecture == 'swem_max':
        endpoints = swem_max_endpoints
    else:
        raise ValueError('No such architecture')

    # CONCEPTS
    print('Getting concepts...')
    data = pd.read_csv(os.path.join(args.data_dir, 'concept_search.csv'), nrows=50000)
    data['text'] = data['text'].apply(clean)
    labels = data['labels'].values

    counter = Counter()
    for t in data['text']:
        counter.update(get_ngrams(t, n=args.ngrams))

    ngram_tokens = [' '.join(a) for a, i in counter.most_common()]
    concept_names = args.concept_names.split()

    concepts = {key: {'pos': None, 'neg': None} for key in concept_names}

    for conc in concept_names:
        concepts[conc]['pos'] = [c for c in ngram_tokens if clean(conc) in c]
        concepts[conc]['neg'] = np.random.choice(ngram_tokens, size=len(concepts[conc]['pos']), replace=False)

    pickle.dump(concepts, open(os.path.join(args.data_dir, 'concepts.pkl'), 'wb'))

    # BOTTLENECKS
    print('Getting bottlenecks...')
    mw = ModelWrapper(args.model_dir, endpoints)

    graph = tf.get_default_graph()

    sess = tf.Session(graph=graph)
    sess.run(tf.global_variables_initializer())
    cav_bottlenecks = dict()

    for key, val in concepts.items():

        X_conc = mw.calculate_bottleneck(sess, concepts[key]['pos'])
        X_rand = mw.calculate_bottleneck(sess, concepts[key]['neg'])
        X = np.append(X_conc, X_rand, axis=0)
        y = np.array([1] * len(X_conc) + [0] * len(X_rand))

        cav_bottlenecks[key] = (X, y)

    pickle.dump(cav_bottlenecks, open(os.path.join(args.data_dir, 'cav_bottlenecks.pkl'), 'wb'))

    # CAVS
    print('Getting CAVs...')
    cavs = dict()

    for key, val in cav_bottlenecks.items():
        cav = CAV()
        v = cav.fit(cav_bottlenecks[key][0], cav_bottlenecks[key][1])  # (X, y)
        cavs[key] = v

    pickle.dump(cavs, open(os.path.join(args.data_dir, 'cavs.pkl'), 'wb'))

    # GRADIENTS
    print('Getting gradients...')
    grads = mw.calculate_grad(sess, labels, data['text'].tolist())
    pickle.dump(grads, open(os.path.join(args.data_dir, 'grads.pkl'), 'wb'))
