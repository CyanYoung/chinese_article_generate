from keras.layers import LSTM, Dense, Dropout, Masking, TimeDistributed


def rnn(embed_input, vocab_num):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    da = Dense(vocab_num, activation='softmax')
    ta = TimeDistributed(da)
    x = Masking()(embed_input)
    x = ra(x)
    x = Dropout(0.2)(x)
    return ta(x)


def cnn(embed_input, vocab_num):
    pass
