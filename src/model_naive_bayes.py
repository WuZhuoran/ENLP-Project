from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import time
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

TRAIN_DATA_FILE = 'train_cleaned.tsv'


def load_train_data(path):
    D = pd.read_csv(path, sep='\t', header=0)

    D['Sentiment'] = D['Sentiment'].map(lambda x: 0 if x == 0 else x)
    D['Sentiment'] = D['Sentiment'].map(lambda x: 1 if x == 2 else x)
    D['Sentiment'] = D['Sentiment'].map(lambda x: 2 if x == 4 else x)

    X_train = D[['Phrase', 'Sentiment']]
    X_train.is_copy = False
    X_train['Phrase'] = X_train['Phrase'].astype(str)

    X = np.array(list(X_train['Phrase']))
    Y_train = np.array(list(D['Sentiment']))

    return X, Y_train


def shuffle_2(a, b):  # Shuffles 2 arrays with the same order
    s = np.arange(a.shape[0])
    np.random.shuffle(s)
    return a[s], b[s]


current_time = time.time()

X_train, Y_train = load_train_data('../data/' + TRAIN_DATA_FILE)

load_time = time.time() - current_time

print('Time to Load ' + TRAIN_DATA_FILE + ': ' + str(load_time) + 's')

# Feature Engineering
Tokenizer = Tokenizer()
Tokenizer.fit_on_texts(X_train)
# Tokenizer.fit_on_texts(X_train)
Tokenizer_vocab_size = len(Tokenizer.word_index) + 1
print("Vocab size", Tokenizer_vocab_size)

# masking
num_test = 1000
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

# padding all text to same size
X_Train_encodedPadded_words = sequence.pad_sequences(encoded_words, maxlen=maxWordCount)
X_Val_encodedPadded_words = sequence.pad_sequences(encoded_words2, maxlen=maxWordCount)

# shuffling the traing Set
shuffle_2(X_Train_encodedPadded_words, Y_train)

current_time = time.time()

classifier = MultinomialNB()
classifier.fit(X_Train_encodedPadded_words, Y_train)

y_pred = classifier.predict(X_Val_encodedPadded_words)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_Val, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
