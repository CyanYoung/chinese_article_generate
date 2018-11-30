from keras.layers import LSTM, Conv1D, Dense, Dropout


win_len = 10


def rnn(embed_input, vocab_num):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    da = Dense(vocab_num, activation='softmax')
    x = ra(embed_input)
    x = Dropout(0.2)(x)
    return da(x)


def cnn(embed_input, vocab_num):
    ca = Conv1D(filters=128, kernel_size=win_len, padding='valid', activation='relu')
    da1 = Dense(200, activation='relu')
    da2 = Dense(vocab_num, activation='softmax')
    x = ca(embed_input)
    x = da1(x)
    x = Dropout(0.2)(x)
    return da2(x)
