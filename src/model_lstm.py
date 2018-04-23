import datetime
import time

import keras
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.constraints import maxnorm
from keras.layers import Embedding
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from gensim.models import KeyedVectors

current_time = time.time()
seed = 19960214
np.random.seed(seed)
TRAIN_DATA_FILE = 'train_mapped.tsv'
TEST_DATA_FILE = 'test.tsv'
model_name = "../data/GoogleNews-vectors-negative300.bin"

# Vector_type = {'Word2vec', 'No'}
vector_type = "No"
train_list = []
test_list = []
word2vec_input = []
num_features = 300


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


def clean_review(raw_review, remove_stopwords=False, output_format="string"):
    """
    Input:
            raw_review: raw text of a movie review
            remove_stopwords: a boolean variable to indicate whether to remove stop words
            output_format: if "string", return a cleaned string
                           if "list", a list of words extracted from cleaned string.
    Output:
            Cleaned string or list.
    """

    # Remove HTML markup
    text = BeautifulSoup(raw_review)

    # Keep only characters
    text = re.sub("[^a-zA-Z]", " ", text.get_text())

    # Split words and store to list
    text = text.lower().split()

    if remove_stopwords:

        # Use set as it has O(1) lookup time
        stops = set(stopwords.words("english"))
        words = [w for w in text if w not in stops]

    else:
        words = text

    # Return a cleaned string or list
    if output_format == "string":
        return " ".join(words)

    elif output_format == "list":
        return words


def review_to_doublelist(review, tokenizer, remove_stopwords=False):
    """
    Function which generates a list of lists of words from a review for word2vec uses.

    Input:
        review: raw text of a movie review
        tokenizer: tokenizer for sentence parsing
                   nltk.data.load('tokenizers/punkt/english.pickle')
        remove_stopwords: a boolean variable to indicate whether to remove stop words

    Output:
        A list of lists.
        The outer list consists of all sentences in a review.
        The inner list consists of all words in a sentence.
    """

    # Create a list of sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    sentence_list = []

    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentence_list.append(clean_review(raw_sentence, False, "list"))
    return sentence_list


def review_to_vec(words, model, num_features):
    """
    Function which generates a feature vector for the given review.

    Input:
        words: a list of words extracted from a review
        model: trained word2vec model
        num_features: dimension of word2vec vectors

    Output:
        a numpy array representing the review
    """

    feature_vec = np.zeros((num_features), dtype="float32")
    word_count = 0

    # index2word is a list consisting of all words in the vocabulary
    # Convert list to set for speed
    index2word_set = set(model.index2word)

    for word in words:
        if word in index2word_set:
            word_count += 1
            feature_vec += model[word]

    feature_vec /= word_count
    return feature_vec


def gen_review_vecs(reviews, model, num_features):
    """
    Function which generates a m-by-n numpy array from all reviews,
    where m is len(reviews), and n is num_feature

    Input:
            reviews: a list of lists.
                     Inner lists are words from each review.
                     Outer lists consist of all reviews
            model: trained word2vec model
            num_feature: dimension of word2vec vectors
    Output: m-by-n numpy array, where m is len(review) and n is num_feature
    """

    curr_index = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:

        if curr_index % 1000 == 0.:
            print(
                "Vectorizing review %d of %d" % (curr_index, len(reviews)))

        review_feature_vecs[curr_index] = review_to_vec(review, model, num_features)
        curr_index += 1

    return review_feature_vecs


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

# Extract words from reviews
print('Start Extract Words from Reviews...')

# xrange is faster when iterating
if vector_type == "Word2vec":
    for i in range(0, len(X_train.Phrase)):
        train_list.append(clean_review(X_train.Phrase[i], output_format="list"))
        if i % 1000 == 0:
            print("Cleaning training review", i)

    for i in range(0, len(X_Val.Phrase)):
        test_list.append(clean_review(X_Val.Phrase[i], output_format="list"))
        if i % 1000 == 0:
            print("Cleaning test review", i)
    print("Loading the pre-trained model")
    model = KeyedVectors.load_word2vec_format(model_name, binary=True)
    model.init_sims(replace=True)
    print("Vectorizing training review")
    train_vec = gen_review_vecs(train_list, model, num_features)
    print("Vectorizing test review")
    test_vec = gen_review_vecs(test_list, model, num_features)
else:

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

# model
model = Sequential()

model.add(Embedding(maxDictionary_size, 32, input_length=maxWordCount))  # to change words to ints
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.5))
# model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# hidden layers
# model.add(Bidirectional(LSTM(10)))
model.add(LSTM(10))
# model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(1200, activation='relu', W_constraint=maxnorm(1)))
# model.add(Dropout(0.6))
model.add(Dense(500, activation='relu', W_constraint=maxnorm(1)))

# model.add(Dropout(0.5))
# output layer
model.add(Dense(5, activation='softmax'))

# Compile model
# adam=Adam(lr=learning_rate, beta_1=0.7, beta_2=0.999, epsilon=1e-08, decay=0.0000001)

model.summary()

learning_rate = 0.0001
epochs = 2
batch_size = 32  # 32
sgd = SGD(lr=learning_rate, nesterov=True, momentum=0.7, decay=1e-4)
Nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=Nadam, metrics=['accuracy'])

tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/log_25', histogram_freq=0, write_graph=True,
                                          write_images=False)
checkpointer = ModelCheckpoint(filepath="./weights/weights_25.hdf5", verbose=1, save_best_only=True, monitor="val_loss")
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=0, verbose=1, mode='auto', cooldown=0,
                              min_lr=1e-6)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

# Loading best weights
# model.load_weights("./weights/weights_19.hdf5")

print("=============================== Training =========================================")

history = model.fit(X_Train_encodedPadded_words, Y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                    validation_data=(X_Val_encodedPadded_words, Y_Val),
                    callbacks=[tensorboard, reduce_lr, checkpointer, earlyStopping])

print("=============================== Score =========================================")

scores = model.evaluate(X_Val_encodedPadded_words, Y_Val, verbose=0)

print('LSTM_model_eta_' +
      str(learning_rate) +
      '_batch_' +
      str(batch_size) +
      '_epochs_' +
      str(epochs) +
      '_layers_' +
      str(3) +
      '_Embedding_' +
      'keras.layers.Embedding' +
      '_NumberTest_' +
      str(num_test) +
      '_timestamp_' +
      str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))

print("Accuracy: %.2f%%" % (scores[1] * 100))

print('Time to Train and Test: ' + str(time.time() - current_time) + 's')
