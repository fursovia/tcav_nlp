"""
New features and data cleaning
"""

import pandas as pd
from collections import Counter
import re
from sklearn.model_selection import train_test_split
import os
import argparse
from model.utils import save_vocab_to_txt_file

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-mf', '--min_freq', type=int, default=5)
parser.add_argument('-s', '--sample', action='store_true')

parser.set_defaults(sample=False)

regex = re.compile(r'[^\w\s]')


def update_vocabulary(tokens, counter):
    counter.update(tokens)


def create_vocab(pd_series, min_count=0):
    vocabulary = Counter()
    _ = pd_series.apply(lambda x: update_vocabulary(x.split(), vocabulary))
    vocabulary = [tok for tok, count in vocabulary.most_common() if count >= min_count]
    vocabulary.insert(0, '<PAD>')
    vocabulary.insert(1, '<UNK>')

    vocab_length = len(vocabulary)
    return vocabulary, vocab_length


def clean(text):
    text = regex.sub(r' ', text).strip()
    text = text.lower()
    return text


if __name__ == '__main__':

    args = parser.parse_args()

    if args.sample:
        nrows = 10000
    else:
        nrows = None

    data = pd.read_csv(os.path.join(args.data_dir, 'dataset.csv'), nrows=nrows)

    data['text'] = data['text'].apply(lambda x: clean(x))

    train, valid = train_test_split(data, stratify=data['labels'], test_size=0.3, random_state=24)

    data.to_csv(os.path.join(args.data_dir, 'full.csv'), index=False)
    train.to_csv(os.path.join(args.data_dir, 'train.csv'), index=False)
    valid.to_csv(os.path.join(args.data_dir, 'eval.csv'), index=False)

    vocabulary, vocab_size = create_vocab(data['text'], min_count=args.min_freq)
    print(f'Vocab size = {vocab_size}')

    save_vocab_to_txt_file(vocabulary, os.path.join(args.data_dir, 'vocab.txt'))
