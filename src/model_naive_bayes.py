import time

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *

TRAIN_DATA_FILE = 'train_mapped.tsv'


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

train_X, test_X, train_y, test_y = train_test_split(train['Phrase'], train['Sentiment'], test_size=0.20)

bigram_vectorizer = CountVectorizer(analyzer="word",
                                    tokenizer=None,
                                    preprocessor=None,
                                    stop_words=None,
                                    ngram_range=(1, 3),
                                    strip_accents='unicode')

bigram_feature_matrix_train = bigram_vectorizer.fit_transform(train_X)
bigram_feature_matrix_test = bigram_vectorizer.transform(test_X)

bigram_multinomialNB_classifier = MultinomialNB().fit(bigram_feature_matrix_train, train_y)
bigram_multinomialNB_prediction = bigram_multinomialNB_classifier.predict(bigram_feature_matrix_test)

model = 'Unigram-Trigram Multinomial Naive Bayes'
target_names = ['0', '1', '2']

print(
    '-------' + '-' * len(model))
print(
    'MODEL:', model)
print(
    '-------' + '-' * len(model))

print(
    'Precision = ' + str(metrics.precision_score(test_y, bigram_multinomialNB_prediction, average=None)))
print(
    'Recall = ' + str(metrics.recall_score(test_y, bigram_multinomialNB_prediction, average=None)))
print(
    'F1 = ' + str(metrics.f1_score(test_y, bigram_multinomialNB_prediction, average=None)))
print(
    'Accuracy = %.2f%%' % (metrics.accuracy_score(test_y, bigram_multinomialNB_prediction) * 100.0))
print(
    'Confusion matrix =  \n' + str(
        metrics.confusion_matrix(test_y, bigram_multinomialNB_prediction, labels=[0, 1, 2])))
print('\nClassification Report:\n' + classification_report(test_y, bigram_multinomialNB_prediction,
                                                           target_names=target_names))
