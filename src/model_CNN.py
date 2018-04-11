import datetime

import keras
import numpy as np
import pandas as pd
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

'''Another config for CNN
backend.set_image_dim_ordering('th')

# Number of feature maps (outputs of convolutional layer)
N_fm = 300
# kernel size of convolutional layer
kernel_size = 8

model = Sequential()
# Embedding layer (lookup table of trainable word vectors)
model.add(Embedding(input_dim=W.shape[0], 
                    output_dim=W.shape[1], 
                    input_length=conv_input_height,
                    weights=[W], 
                    W_constraint=unitnorm()))
# Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
model.add(Reshape((1, conv_input_height, conv_input_width)))

# first convolutional layer
model.add(Convolution2D(N_fm, 
                        kernel_size, 
                        conv_input_width, 
                        border_mode='valid', 
                        W_regularizer=l2(0.0001)))
# ReLU activation
model.add(Activation('relu'))

# aggregate data in every feature map to scalar using MAX operation
model.add(MaxPooling2D(pool_size=(conv_input_height-kernel_size+1, 1)))

model.add(Flatten())
model.add(Dropout(0.5))
# Inner Product layer (as in regular neural network, but without non-linear activation function)
model.add(Dense(2))
# SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
model.add(Activation('softmax'))

# Custom optimizers could be used, though right now standard adadelta is employed
opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
model.compile(loss='categorical_crossentropy', 
              optimizer=opt,
              metrics=['accuracy'])
              
epoch = 0
val_acc = []
val_auc = []

N_epoch = 3

for i in xrange(N_epoch):
    model.fit(train_X, train_Y, batch_size=50, nb_epoch=1, verbose=1)
    output = model.predict_proba(val_X, batch_size=10, verbose=1)
    # find validation accuracy using the best threshold value t
    vacc = np.max([np.sum((output[:,1]>t)==(val_Y[:,1]>0.5))*1.0/len(output) for t in np.arange(0.0, 1.0, 0.01)])
    # find validation AUC
    vauc = roc_auc_score(val_Y, output)
    val_acc.append(vacc)
    val_auc.append(vauc)
    print 'Epoch {}: validation accuracy = {:.3%}, validation AUC = {:.3%}'.format(epoch, vacc, vauc)
    epoch += 1
    
print '{} epochs passed'.format(epoch)
print 'Accuracy on validation dataset:'
print val_acc
print 'AUC on validation dataset:'
print val_auc
'''

# Final evaluation of the model
scores = model.evaluate(X_Val_encodedPadded_words, Y_Val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
