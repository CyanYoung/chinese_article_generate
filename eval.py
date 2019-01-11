import numpy as np

from generate import predict

from util import flat_read


path_test = 'data/test.csv'
texts = flat_read(path_test, 'text')


def test(name, texts):
    probs = predict(texts, name)
    preds = np.argmax(probs, axis=1)


if __name__ == '__main__':
    test('rnn', texts)
    test('cnn', texts)
