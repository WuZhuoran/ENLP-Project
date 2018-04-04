from textblob import TextBlob
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn
from sklearn.metrics import accuracy_score


def load_train_data(path):  # loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path, sep='\t', header=0)

    D['Sentiment'] = D['Sentiment'].map(lambda x: 0 if x == 0 else x)
    D['Sentiment'] = D['Sentiment'].map(lambda x: 1 if x == 2 else x)
    D['Sentiment'] = D['Sentiment'].map(lambda x: 2 if x == 4 else x)

    X_train = D[['Phrase', 'Sentiment']]

    return X_train


X_train = load_train_data('../data/train_extract.tsv')

X_train['text_len'] = X_train['Phrase'].apply(len)
X_train['Phrase'] = X_train['Phrase'].apply(lambda x: str(x).strip())

X_train['text_blob'] = X_train['Phrase'].map(lambda x: TextBlob(x).sentiment)
X_train['polarity'] = X_train['text_blob'].map(lambda x: x[0])
X_train['subjectivity'] = X_train['text_blob'].map(lambda x: x[1])

cols = ['text_len', 'polarity', 'subjectivity']

params = {
    'eta': 0.002,
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 6,
    'eval_metric': 'mlogloss',
    'seed': 2017,
    'silent': True
}

x1, x2, y1, y2 = sklearn.model_selection.train_test_split(X_train[cols], X_train['Sentiment'], test_size=0.2,
                                                          random_state=2017)
dtrain = xgb.DMatrix(x1, label=y1)
dvalid = xgb.DMatrix(x2, label=y2)

watchlist = [(dvalid, 'valid'), (dtrain, 'train')]

model = xgb.train(params, dtrain, 100, watchlist, verbose_eval=5, early_stopping_rounds=5)

y_pred = model.predict(dvalid)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y2, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
