from keras.layers import LSTM, Dense, Dropout, TimeDistributed


def rnn_plain(embed_input, vocab_num):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    da = Dense(vocab_num, activation='softmax')
    ta = TimeDistributed(da)
    x = ra(embed_input)
    x = Dropout(0.5)(x)
    return ta(x)


def rnn_stack(embed_input, vocab_num):
    ra1 = LSTM(200, activation='tanh', return_sequences=True)
    ra2 = LSTM(200, activation='tanh', return_sequences=True)
    da = Dense(vocab_num, activation='softmax')
    ta = TimeDistributed(da)
    x = ra1(embed_input)
    x = ra2(x)
    x = Dropout(0.5)(x)
    return ta(x)
