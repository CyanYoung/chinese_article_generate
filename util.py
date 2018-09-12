import pandas as pd


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


def map_path(name, paths):
    if name in paths:
        return paths[name]
    else:
        raise KeyError


def map_func(name, funcs):
    if name in funcs:
        return funcs[name]
    else:
        raise KeyError


def map_model(name, models):
    if name in models:
        return models[name]
    else:
        raise KeyError
