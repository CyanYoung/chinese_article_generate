import pickle as pk

import numpy as np
from numpy.random import choice

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from util import map_path


seq_len = 20
max_len = 100

models = {'rnn_plain': load_model(map_path('rnn_plain')),
          'rnn_stack': load_model(map_path('rnn_stack'))}


path_word2ind = 'model/word2ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
word_inds = word2ind.word_index

puncs = ['，', '。', '#']
punc_inds = [word_inds[punc] for punc in puncs]

ind_words = dict()
for word, ind in word_inds.items():
    ind_words[ind] = word


def map_model(name):
    if name in models:
        return models[name]
    else:
        raise KeyError


def sample(probs, ind_words):
    max_probs = np.array(sorted(probs, reverse=True)[:10])
    max_probs = max_probs / np.sum(max_probs)
    max_inds = np.argsort(-probs)[:10]
    if max_inds[0] in punc_inds:
        next_ind = max_inds[0]
    else:
        next_ind = choice(max_inds, p=max_probs)
    return ind_words[next_ind]


def predict(sent, name):
    next_word = ''
    while next_word != '#' and len(sent) < max_len:
        seq = word2ind.texts_to_sequences([sent])[0]
        align_seq = pad_sequences([seq], maxlen=seq_len)
        model = map_model(name)
        probs = model.predict(align_seq)[0]
        next_word = sample(probs, ind_words)
        sent = sent + next_word
    return sent


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('plain: %s' % predict(text, 'rnn_plain'))
        print('stack: %s' % predict(text, 'rnn_stack'))
