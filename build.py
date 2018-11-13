import pickle as pk

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from nn_arch import rnn_plain, rnn_stack

from util import map_item


batch_size = 512

path_embed = 'feat/embed.pkl'
path_align_seq = 'feat/align_seq.pkl'
path_next_ind = 'feat/next_ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_align_seq, 'rb') as f:
    align_seqs = pk.load(f)
with open(path_next_ind, 'rb') as f:
    next_inds = pk.load(f)

funcs = {'rnn_plain': rnn_plain,
         'rnn_stack': rnn_stack}

paths = {'rnn_plain': 'model/rnn_plain.h5',
         'rnn_stack': 'model/rnn_stack.h5'}


def compile(name, embed_mat, seq_len):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True)
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    func = map_item(name, funcs)
    output = func(embed_input, vocab_num)
    model = Model(input, output)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def split(rate, align_seqs, next_inds, vocab_num):
    bound = int(len(align_seqs) * rate)
    seq_train, ind_train = align_seqs[:bound], next_inds[:bound]
    x_dev = np.array(align_seqs[bound:])
    y_dev = to_categorical(next_inds[bound:], num_classes=vocab_num)
    return seq_train, ind_train, x_dev, y_dev


def get_part(l_rate, step, seq_train, ind_train, vocab_num):
    l_bound = int(len(seq_train) * l_rate)
    u_bound = int(len(seq_train) * (l_rate + step))
    x_train = np.array(seq_train[l_bound:u_bound])
    y_train = to_categorical(ind_train[l_bound:u_bound], num_classes=vocab_num)
    return x_train, y_train


def fit(name, epoch, embed_mat, align_seqs, next_inds, step):
    vocab_num, embed_len = embed_mat.shape
    seq_len = len(align_seqs[0])
    model = compile(name, embed_mat, seq_len)
    check_point = ModelCheckpoint(map_item(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    seq_train, ind_train, x_dev, y_dev = split(0.8, align_seqs, next_inds, vocab_num)
    for l_rate in np.arange(0, 1, step):
        x_train, y_train = get_part(l_rate, step, seq_train, ind_train, vocab_num)  # limit memory
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch,
                  verbose=True, callbacks=[check_point], validation_data=(x_dev, y_dev))


if __name__ == '__main__':
    fit('rnn_plain', 10, embed_mat, align_seqs, next_inds, step=0.2)
    fit('rnn_stack', 10, embed_mat, align_seqs, next_inds, step=0.2)
