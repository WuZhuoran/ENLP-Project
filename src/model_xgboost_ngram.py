import datetime
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import *
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

TRAIN_DATA_FILE = 'train_cleaned.tsv'


def load_train_data(path):
    data = pd.read_csv(path, sep='\t', header=0)

    data['Sentiment'] = data['Sentiment'].map(lambda x: 0 if x == 0 else x)
    data['Sentiment'] = data['Sentiment'].map(lambda x: 1 if x == 2 else x)
    data['Sentiment'] = data['Sentiment'].map(lambda x: 2 if x == 4 else x)

    # Remove empty
    data['Phrase'].replace('', np.nan, inplace=True)
    data.dropna(subset=['Phrase'], inplace=True)

    data['Phrase'] = data['Phrase'].astype(str)

    return data


current_time = time.time()

train = load_train_data('../data/' + TRAIN_DATA_FILE)

load_time = time.time() - current_time

print('Time to Load ' + TRAIN_DATA_FILE + ': ' + str(load_time) + 's')

test_size = 0.2

train_X, test_X, train_y, test_y = train_test_split(train['Phrase'], train['Sentiment'], test_size=test_size)

bigram_vectorizer = CountVectorizer(analyzer="word",
                                    tokenizer=None,
                                    preprocessor=None,
                                    stop_words=None,
                                    ngram_range=(1, 3),
                                    strip_accents='unicode')

bigram_feature_matrix_train = bigram_vectorizer.fit_transform(train_X)
bigram_feature_matrix_test = bigram_vectorizer.transform(test_X)

current_time = time.time()
num_round = 2000

params = {
    'eta': 0.002,
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 6,
    'eval_metric': 'mlogloss',
    'seed': 2017,
    'silent': True
}

dtrain = xgb.DMatrix(bigram_feature_matrix_train, label=train_y)
dvalid = xgb.DMatrix(bigram_feature_matrix_test, label=test_y)

watchlist = [(dvalid, 'valid'), (dtrain, 'train')]
target_names = ['0', '1', '2']

model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval=5, early_stopping_rounds=5)

y_pred = model.predict(dvalid)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(test_y, predictions)

print('XGBoost_model_eta_' +
      str(params['eta']) +
      '_round_' +
      str(num_round) +
      '_NumberFeatures_' +
      'Unigram-Trigram Model' +
      '_TestSize_' +
      str(test_size) +
      '_timestamp_' +
      str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))

print(
    'Precision = ' + str(metrics.precision_score(test_y, predictions, average=None)))
print(
    'Recall = ' + str(metrics.recall_score(test_y, predictions, average=None)))
print(
    'F1 = ' + str(metrics.f1_score(test_y, predictions, average=None)))
print(
    'Accuracy = %.2f%%' % (metrics.accuracy_score(test_y, predictions) * 100.0))
print(
    'Confusion matrix =  \n' + str(
        metrics.confusion_matrix(test_y, predictions, labels=[0, 1, 2])))
print('\nClassification Report:\n' + classification_report(test_y, predictions,
                                                           target_names=target_names))

print('Time to Train and Test: ' + str(time.time() - current_time) + 's')
