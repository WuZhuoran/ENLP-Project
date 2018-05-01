import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

TRAIN_DATA_FILE = 'train_extract.tsv'


def review_to_words(raw_review):
    review = raw_review
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return ' '.join(review)


nltk.download('stopwords')

train = pd.read_csv('../data/' + TRAIN_DATA_FILE, sep='\t', header=0)

corpus = []
for i in range(0, 8529):
    corpus.append(review_to_words(train['Phrase'][i]))

train['Phrase'] = corpus

train.to_csv('../data/train_cleaned.tsv', sep='\t', index=False)
