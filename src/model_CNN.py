import datetime

import keras
import numpy as np
import pandas as pd
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence

seed = 19960214
np.random.seed(seed)
TRAIN_DATA_FILE = 'train_cleaned.tsv'
TEST_DATA_FILE = 'test.tsv'


def load_train_data(path):  # loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path, sep='\t', header=0)
    feature_names = np.array(list(D.columns.values))
    X_train = np.array(list(D['Phrase']))
    Y_train = np.array(list(D['Sentiment']))
    return X_train, Y_train, feature_names


def load_test_data(path):  # loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path, sep='\t', header=0)
    X_test = np.array(list(D['Phrase']))
    X_test_PhraseID = np.array(list(D['PhraseId']))
    return X_test, X_test_PhraseID


def shuffle_2(a, b):  # Shuffles 2 arrays with the same order
    s = np.arange(a.shape[0])
    np.random.shuffle(s)
    return a[s], b[s]


X_train, Y_train, feature_names = load_train_data('../data/' + TRAIN_DATA_FILE)
X_test, X_test_PhraseID = load_test_data('../data/' + TEST_DATA_FILE)

print('============================== Training data shapes ==============================')
print('X_train.shape is ', X_train.shape)
print('Y_train.shape is ', Y_train.shape)

Tokenizer = Tokenizer()
Tokenizer.fit_on_texts(np.concatenate((X_train, X_test), axis=0))
# Tokenizer.fit_on_texts(X_train)
Tokenizer_vocab_size = len(Tokenizer.word_index) + 1
print("Vocab size", Tokenizer_vocab_size)

# masking
num_test = int(0.2 * len(X_train))
mask = range(num_test)

Y_Val = Y_train[:num_test]
Y_Val2 = Y_train[:num_test]
X_Val = X_train[:num_test]

X_train = X_train[num_test:]
Y_train = Y_train[num_test:]

maxWordCount = 60
maxDictionary_size = Tokenizer_vocab_size

encoded_words = Tokenizer.texts_to_sequences(X_train)
encoded_words2 = Tokenizer.texts_to_sequences(X_Val)
encoded_words3 = Tokenizer.texts_to_sequences(X_test)

# padding all text to same size
X_Train_encodedPadded_words = sequence.pad_sequences(encoded_words, maxlen=maxWordCount)
X_Val_encodedPadded_words = sequence.pad_sequences(encoded_words2, maxlen=maxWordCount)
X_test_encodedPadded_words = sequence.pad_sequences(encoded_words3, maxlen=maxWordCount)

# One Hot Encoding
Y_train = keras.utils.to_categorical(Y_train, 5)
Y_Val = keras.utils.to_categorical(Y_Val, 5)

# shuffling the traing Set
shuffle_2(X_Train_encodedPadded_words, Y_train)

embedding_vecor_length = 32

model = Sequential()
model.add(Embedding(maxDictionary_size, embedding_vecor_length, input_length=maxWordCount))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_Train_encodedPadded_words, Y_train, epochs=3, batch_size=32)

# Final evaluation of the model
scores = model.evaluate(X_Val_encodedPadded_words, Y_Val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

