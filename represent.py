import pickle as pk

import numpy as np

from gensim.models.word2vec import Word2Vec

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from util import flat_read, add_flag


embed_len = 200
min_freq = 50
max_vocab = 3000
seq_len = 20

path_word_vec = 'feat/word_vec.pkl'
path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'


def word2vec(sents, path_word_vec):
    model = Word2Vec(sents, size=embed_len, window=3, min_count=min_freq, negative=5, iter=10)
    word_vecs = model.wv
    with open(path_word_vec, 'wb') as f:
        pk.dump(word_vecs, f)
    if __name__ == '__main__':
        words = ['，', '。', '*', '#']
        for word in words:
            print(word_vecs.most_similar(word))


def embed(sents, path_word2ind, path_word_vec, path_embed):
    model = Tokenizer(num_words=max_vocab, filters='', char_level=True)
    model.fit_on_texts(sents)
    word_inds = model.word_index
    with open(path_word2ind, 'wb') as f:
        pk.dump(model, f)
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    vocab = word_vecs.vocab
    vocab_num = min(max_vocab, len(word_inds))
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def align(sents, path_word2ind, path_align_seq, path_next_ind):
    with open(path_word2ind, 'rb') as f:
        model = pk.load(f)
    seqs = model.texts_to_sequences(sents)
    align_seqs = list()
    next_inds = list()
    for seq in seqs:
        for u_bound in range(1, len(seq)):
            align_seq = pad_sequences([seq[:u_bound]], maxlen=seq_len)[0]
            align_seqs.append(align_seq)
            next_inds.append(seq[u_bound])
    with open(path_align_seq, 'wb') as f:
        pk.dump(align_seqs, f)
    with open(path_next_ind, 'wb') as f:
        pk.dump(next_inds, f)


def vectorize(path_train, path_align_seq, path_next_ind):
    texts = flat_read(path_train, 'text')
    sents = add_flag(texts)
    # word2vec(sents, path_word_vec)
    # embed(sents, path_word2ind, path_word_vec, path_embed)
    align(sents, path_word2ind, path_align_seq, path_next_ind)


if __name__ == '__main__':
    path_train = 'data/train.csv'
    path_align_seq = 'feat/align_seq.pkl'
    path_next_ind = 'feat/next_ind.pkl'
    vectorize(path_train, path_align_seq, path_next_ind)
