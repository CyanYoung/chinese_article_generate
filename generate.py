import pickle as pk

import numpy as np
from numpy.random import choice

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from util import map_item


seq_len = 20
min_len = 20
max_len = 100
bos = '*'
eos = '#'

path_word2ind = 'model/word2ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
word_inds = word2ind.word_index

puncs = ['，', '。']
punc_inds = [word_inds[punc] for punc in puncs]

ind_words = dict()
for word, ind in word_inds.items():
    ind_words[ind] = word

paths = {'rnn_plain': 'model/rnn_plain.h5',
         'rnn_stack': 'model/rnn_stack.h5'}

models = {'rnn_plain': load_model(map_item('rnn_plain', paths)),
          'rnn_stack': load_model(map_item('rnn_stack', paths))}


def sample(probs, sent_len, word_inds, ind_words):
    max_probs = np.array(sorted(probs, reverse=True)[:10])
    max_probs = max_probs / np.sum(max_probs)
    max_inds = np.argsort(-probs)[:10]
    if max_inds[0] in punc_inds:
        next_ind = max_inds[0]
    elif sent_len < min_len:
        next_ind = word_inds[eos]
        while next_ind == word_inds[eos]:
            next_ind = choice(max_inds, p=max_probs)
    else:
        next_ind = choice(max_inds, p=max_probs)
    return ind_words[next_ind]


def predict(text, name):
    sent = text.strip()
    if len(sent) < 1 or sent[0] != bos:
        sent = bos + sent
    next_word = ''
    while next_word != eos and len(sent) < max_len:
        sent = sent + next_word
        seq = word2ind.texts_to_sequences([sent])[0]
        align_seq = pad_sequences([seq], maxlen=seq_len)
        model = map_item(name, models)
        probs = model.predict(align_seq)[0]
        next_word = sample(probs, len(sent), word_inds, ind_words)
    return sent[1:]


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('plain: %s' % predict(text, 'rnn_plain'))
        print('stack: %s' % predict(text, 'rnn_stack'))
