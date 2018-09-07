import pandas as pd

from nn_arch import rnn_plain, rnn_stack


def flat_read(path_train, field):
    nest_items = pd.read_csv(path_train, usecols=[field], keep_default_na=False).values
    items = list()
    for nest_item in nest_items:
        items.append(nest_item[0])
    return items


def add_flag(sents):
    flag_sents = list()
    for sent in sents:
        flag_sent = '*' + sent + '#'
        flag_sents.append(flag_sent)
    return flag_sents


funcs = {'rnn_plain': rnn_plain,
         'rnn_stack': rnn_stack}

paths = {'rnn_plain': 'model/rnn_plain.h5',
         'rnn_stack': 'model/rnn_stack.h5'}


def map_path(name):
    if name in paths:
        return paths[name]
    else:
        raise KeyError


def map_func(name):
    if name in funcs:
        return funcs[name]
    else:
        raise KeyError
