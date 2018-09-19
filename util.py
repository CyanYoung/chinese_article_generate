import pandas as pd


def flat_read(path, field):
    nest_items = pd.read_csv(path, usecols=[field], keep_default_na=False).values
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


def map_item(name, items):
    if name in items:
        return items[name]
    else:
        raise KeyError
