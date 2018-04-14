from IPython.display import display, HTML

import logging
import os
import re
import pandas as pd
import numpy as np # linear algebra
import seaborn as sn # To get nice plots
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=10)

# Dependecy imports
import pandas as pd


def _parse(data):

    data['Phrase'] = data['Phrase'].map(lambda x: x.lower()) # Convert all letters to lower case
    data['Phrase'] = data['Phrase'].map((lambda x: re.sub(r'[^a-zA-z0-9\s]', '', x)))

    return data


def _load():
    # Load datasets
    train = pd.read_csv("../data/train.tsv", header=0, delimiter="\t", quoting=3,
                        dtype={'Sentiment': 'category'})
    test = pd.read_csv("../data/test.tsv", header=0, delimiter="\t", quoting=3)

    logging.info('Train, number of phrases: %d', train["PhraseId"].size)
    logging.info('Test, number of phrases: %d', test["PhraseId"].size)

    train = _parse(train)[["Phrase", "Sentiment"]]
    test = _parse(test)[["PhraseId", "Phrase"]]

    return train, test


TRAIN_SET, TEST_SET = _load()

'''
The sentiment labels are:
0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive
'''

# Number of Sentiment classes
TRAIN_SET['Sentiment'].value_counts().plot(kind='bar', title='Number of Sentiment classes')
plt.show()

# TRAIN_SET[TRAIN_SET['Sentiment'] != '2'].head(20)

# Number of words per Phrase in both datasets
TRAIN_SET['Phrase'].str.split().str.len().value_counts().plot(kind='bar', figsize=(16, 6))
plt.show()

TEST_SET['Phrase'].str.split().str.len().value_counts().plot(kind='bar', figsize=(16, 6))
plt.show()

# 20 most popular words in boths sets
unique_words_train = TRAIN_SET['Phrase'].str.split(' ', expand=True).stack()

print('Unique words:', unique_words_train.unique().shape[0])
print(unique_words_train.value_counts().head(30))

unique_words_test = TEST_SET['Phrase'].str.split(' ', expand=True).stack()

print('Unique words:', unique_words_test.unique().shape[0])
print(unique_words_test.value_counts().head(30))

# Words which are in Test set but not in Train set
not_in_train = unique_words_test.unique()[~np.in1d(unique_words_test.unique(), unique_words_train.unique())]

print('Words in Test but not in Train', not_in_train.shape[0])
print(unique_words_test[unique_words_test.isin(not_in_train)].value_counts().head(30))

# Words which are in Ttra set but not in Test set
not_in_test = unique_words_train.unique()[~np.in1d(unique_words_train.unique(), unique_words_test.unique())]

print('Words in Test but not in Train', not_in_test.shape[0])
print(unique_words_train[unique_words_train.isin(not_in_test)].value_counts().head(30))
