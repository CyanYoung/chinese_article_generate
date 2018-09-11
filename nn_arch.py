from keras.layers import LSTM, Dense, Masking, Dropout


def rnn_plain(embed_input, vocab_num):
    ra = LSTM(300, activation='tanh')
    da = Dense(vocab_num, activation='softmax')
    x = Masking()(embed_input)
    x = ra(x)
    x = Dropout(0.2)(x)
    return da(x)


def rnn_stack(embed_input, vocab_num):
    ra1 = LSTM(300, activation='tanh', return_sequences=True)
    ra2 = LSTM(300, activation='tanh')
    da = Dense(vocab_num, activation='softmax')
    x = Masking()(embed_input)
    x = ra1(x)
    x = ra2(x)
    x = Dropout(0.2)(x)
    return da(x)
