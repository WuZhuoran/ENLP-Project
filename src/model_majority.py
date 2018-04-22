import datetime
import time

import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.metrics import *
from textblob import TextBlob

TRAIN_DATA_FILE = 'train_cleaned.tsv'


def load_train_data(path):
    D = pd.read_csv(path, sep='\t', header=0)

    D['Sentiment'] = D['Sentiment'].map(lambda x: 0 if x == 0 else x)
    D['Sentiment'] = D['Sentiment'].map(lambda x: 1 if x == 2 else x)
    D['Sentiment'] = D['Sentiment'].map(lambda x: 2 if x == 4 else x)

    X_train = D[['Phrase', 'Sentiment']]
    X_train.is_copy = False
    X_train['Phrase'] = X_train['Phrase'].astype(str)

    return X_train


current_time = time.time()

X_train = load_train_data('../data/' + TRAIN_DATA_FILE)

load_time = time.time() - current_time

print('Time to Load ' + TRAIN_DATA_FILE + ': ' + str(load_time) + 's')

# Feature Engineering
X_train['text_blob'] = X_train['Phrase'].map(lambda x: TextBlob(x).sentiment)
X_train['polarity'] = X_train['text_blob'].map(lambda x: x[0])
X_train['subjectivity'] = X_train['text_blob'].map(lambda x: x[1])

cols = ['polarity', 'subjectivity']

current_time = time.time()
test_size = 0.2

x1, x2, y1, y2 = sklearn.model_selection.train_test_split(X_train[cols], X_train['Sentiment'], test_size=test_size,
                                                          random_state=19960214)

target_names = ['0', '1', '2']

model = DummyClassifier(strategy='most_frequent', random_state=0)
model.fit(x1, y1)

y_pred = model.predict(x2)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y2, predictions)

print('Major_class_model_eta_' +
      '_TestSize_' +
      str(test_size) +
      '_timestamp_' +
      str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))

print(
    'Accuracy = %.2f%%' % (metrics.accuracy_score(y2, predictions) * 100.0))

print('Time to Train and Test: ' + str(time.time() - current_time) + 's')
