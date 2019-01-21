# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.optimizers import RMSprop
from sentenses import prepare_train_data, generate_sentense
import numpy as np

size = 10
xs, ys, word_index, index_word = prepare_train_data(sentence_size=size)
# xs = np.reshape(xs, (xs.shape[0], 1, xs.shape[1]))

model = Sequential()

model.add(Embedding(len(word_index), 96))
model.add(Dropout(0.15))
model.add(LSTM(96))
model.add(Dropout(0.15))
model.add(Dense(len(word_index), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop())
model.fit(xs, ys, epochs=100, verbose=1, batch_size=256)


def fun(model, word_index, index_word, n=30, sentense_size=2):
    result = generate_sentense(model, word_index, index_word,
                               n, sentense_size)
    for i in result:
        print(i)


fun(model, word_index, index_word, n=10, sentense_size=size)
