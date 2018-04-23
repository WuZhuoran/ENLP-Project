import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
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

stop = set(STOPWORDS)

wordcloud = WordCloud(background_color='white',
                      stopwords=stop,
                      max_words=50,
                      max_font_size=80,
                      random_state=42
                      ).generate(' '.join(train['Phrase']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
