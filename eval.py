import pickle as pk

import numpy as np

from keras.models import load_model

from util import map_item


path_rnn_sent = 'feat/rnn_sent_train.pkl'
path_cnn_sent = 'feat/cnn_sent_train.pkl'
path_label = 'feat/label.pkl'
path_word2ind = 'model/word2ind.pkl'
with open(path_rnn_sent, 'rb') as f:
    rnn_sents = pk.load(f)
with open(path_cnn_sent, 'rb') as f:
    cnn_sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)

word_inds = word2ind.word_index

paths = {'rnn': 'model/rnn.h5',
         'cnn': 'model/cnn.h5'}

models = {'rnn': load_model(map_item('rnn', paths)),
          'cnn': load_model(map_item('cnn', paths))}


def test(name, texts, labels):
    model = map_item(name, models)
    probs = model.predict(texts, name, 'search')
    preds = np.argmax(probs, axis=1)


if __name__ == '__main__':
    test('rnn', rnn_sents, labels)
    test('cnn', cnn_sents, labels)
