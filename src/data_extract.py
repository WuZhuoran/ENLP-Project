import pandas as pd

train = pd.read_csv('../data/train_mapped.tsv', sep='\t', header=0)

data = pd.DataFrame(columns=['SentenceId','Phrase', 'Sentiment'])

temp = list(train['SentenceId'])

count = 1

for index, row in train.iterrows():
    if row['SentenceId'] == count:
        data = data.append(row[['SentenceId', 'Phrase', 'Sentiment']])
        count += 1
        # if count == 2628 or count == 2746 or count == 4044 or count == 4365:
        #    count += 1
        if count not in temp:
            print(count)
            count += 1

data = data.reset_index()
data = data.drop('index', axis=1)

print(len(data))

data.to_csv('../data/train_extract.tsv', sep='\t', index=False)
