"""
utility functions
"""

import json
import re
import yaml
from sklearn.metrics import f1_score
import numpy as np

regex = re.compile(r'[^\w\s]')


def get_yaml_config(config_path):

    with open(config_path, 'r', encoding='utf-8') as f:
        params = yaml.load(f)

    return params


def save_dict_to_yaml(d, yaml_path):
    with open(yaml_path, 'w') as file:
        yaml.dump(d, file, default_flow_style=False)


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4)


def save_vocab_to_txt_file(vocab, txt_path):
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))


def calculate_metrics(probs, labels):

    y_pred = np.argmax(probs, axis=1)

    metrics = dict()
    metrics['f1'] = f1_score(y_true=labels, y_pred=y_pred)

    return metrics


def clean(text):
    text = regex.sub(r' ', text).strip()
    text = re.sub(r' +', ' ', text)
    text = text.lower()
    return text


def vectorize(text, seq_len, vocab, emb_matrix):
    text = clean(text)
    tokens = text.split()

    vectorized = [0] * seq_len  # <PAD>
    for i, tok in enumerate(tokens):
        try:
            tok_id = vocab[tok]
        except KeyError:
            tok_id = 1  # <UNK>

        vectorized[i] = tok_id

    return emb_matrix[vectorized, :]


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)]).reshape(-1, num_classes)
