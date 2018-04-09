import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import *
from sklearn import metrics
from textblob import TextBlob
import time
import datetime


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
num_round = 2000
test_size = 0.2

params = {
    'eta': 0.002,
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 6,
    'eval_metric': 'mlogloss',
    'seed': 2017,
    'silent': True
}

x1, x2, y1, y2 = sklearn.model_selection.train_test_split(X_train[cols], X_train['Sentiment'], test_size=test_size,
                                                          random_state=19960214)
dtrain = xgb.DMatrix(x1, label=y1)
dvalid = xgb.DMatrix(x2, label=y2)

watchlist = [(dvalid, 'valid'), (dtrain, 'train')]
target_names = ['0', '1', '2']

model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval=5, early_stopping_rounds=5)

y_pred = model.predict(dvalid)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y2, predictions)

print('XGBoost_model_eta_' +
      str(params['eta']) +
      '_round_' +
      str(num_round) +
      '_NumberFeatures_' +
      str(len(cols)) +
      '_TestSize_' +
      str(test_size) +
      '_timestamp_' +
      str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")))

print(
    'Precision = ' + str(metrics.precision_score(y2, predictions, average=None)))
print(
    'Recall = ' + str(metrics.recall_score(y2, predictions, average=None)))
print(
    'F1 = ' + str(metrics.f1_score(y2, predictions, average=None)))
print(
    'Accuracy = %.2f%%' % (metrics.accuracy_score(y2, predictions) * 100.0))
print(
    'Confusion matrix =  \n' + str(
        metrics.confusion_matrix(y2, predictions, labels=[0, 1, 2])))
print('\nClassification Report:\n' + classification_report(y2, predictions,
                                                           target_names=target_names))

print('Time to Train and Test: ' + str(time.time() - current_time) + 's')
