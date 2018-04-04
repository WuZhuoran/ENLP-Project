import pandas as pd

train = pd.read_csv('../data/train.tsv', sep='\t', header=0)

# Map 1 to 0, 3 to 4

train['Sentiment'] = train['Sentiment'].map(lambda x: 0 if x == 1 else x)
train['Sentiment'] = train['Sentiment'].map(lambda x: 4 if x == 3 else x)

train.to_csv('../data/train_mapped.tsv', sep='\t', index=False)
