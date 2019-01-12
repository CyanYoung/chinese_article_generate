import pickle as pk

import numpy as np

from keras.models import load_model

from util import map_item


seq_len = 100

path_rnn_sent = 'feat/rnn_sent_test.pkl'
path_cnn_sent = 'feat/cnn_sent_test.pkl'
path_label = 'feat/label_test.pkl'
with open(path_rnn_sent, 'rb') as f:
    rnn_sents = pk.load(f)
with open(path_cnn_sent, 'rb') as f:
    cnn_sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)

paths = {'rnn': 'model/rnn.h5',
         'cnn': 'model/cnn.h5'}

models = {'rnn': load_model(map_item('rnn', paths)),
          'cnn': load_model(map_item('cnn', paths))}


def test(name, sents, labels):
    model = map_item(name, models)
    probs = model.predict(sents)
    len_sum, log_sum = [0] * 2
    for sent, label, prob in zip(sents, labels, probs):
        bound = sum(sent == 0)
        len_sum = len_sum + seq_len - bound
        sent_log = 0
        for i in range(bound, seq_len):
            sent_log = sent_log + np.log(prob[i][label[i]])
        log_sum = log_sum + sent_log
    print('\n%s %s %.2f' % (name, 'perp:', np.power(2, -log_sum / len_sum)))


if __name__ == '__main__':
    test('rnn', rnn_sents, labels)
    test('cnn', cnn_sents, labels)
