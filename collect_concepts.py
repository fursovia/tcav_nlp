import os
import argparse
import pandas as pd
import numpy as np
from nltk import ngrams
from collections import Counter
import pickle
from model.utils import clean

parser = argparse.ArgumentParser()

parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-c', '--concept_names', default='путин россия сша')
parser.add_argument('--ngrams', type=int, default=2)


def get_ngrams(text, n=2):
    words = text.split()
    ngram_tokens = ngrams(words, n)
    return list(ngram_tokens)


if __name__ == '__main__':
    args = parser.parse_args()

    data = pd.read_csv(os.path.join(args.data_dir, 'concept_search.csv'), nrows=50000)

    data['text'] = data['text'].apply(clean)

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
