from textblob import TextBlob
import sklearn
import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

train = pd.read_csv('train.tsv', delimiter='\t', header=0)

train['Sentiment'] = train['Sentiment'].map(lambda x: 'neg' if x == 0 or x == 1 else x)
train['Sentiment'] = train['Sentiment'].map(lambda x: 'neu' if x == 2 else x)
train['Sentiment'] = train['Sentiment'].map(lambda x: 'pos' if x == 3 or x == 4 else x)

print(type(train))

sentiment_pred = []

for phrase in train['Phrase']:
    blob = TextBlob(phrase)
    if blob.sentiment.subjectivity <= 0.5:
        sentiment_pred.append('neu')
    elif blob.sentiment.polarity < 0:
        sentiment_pred.append('neg')
    else:
        sentiment_pred.append('pos')

print(sklearn.metrics.confusion_matrix(train['Sentiment'], sentiment_pred, labels=['neu', 'neg', 'pos']))
print(sklearn.metrics.classification_report(train['Sentiment'], sentiment_pred, labels=['neu', 'neg', 'pos']))

# for row in train.itertuples():
#     print(row)


"""    
text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''

blob = TextBlob(text)
blob.tags  # [('The', 'DT'), ('titular', 'JJ'), ('threat', 'NN'), ('of', 'IN'), ...]

blob.noun_phrases   # WordList(['titular threat', 'blob', 'ultimate movie monster', 'amoeba-like mass', ...])

for sentence in blob.sentences:
    print(sentence.sentiment.polarity)
# 0.060
# -0.341

blob.translate(to="es")  # 'La amenaza titular de The Blob...'
"""