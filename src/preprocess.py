import pandas as pd
import json
from sklearn.model_selection import train_test_split

def preprocess_chexpert(csv_path, out_train, out_val, seed=42):
    df = pd.read_csv(csv_path)
    df.drop_duplicates(subset='id', inplace=True)
    df['Sex'].fillna('Unknown', inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df = pd.get_dummies(df, columns=['Sex','Frontal/Lateral','AP/PA'])
    df['Age_norm'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()

    ids = df['id'].unique()
    train_ids, val_ids = train_test_split(ids, test_size=0.15, random_state=seed)

    for split_ids, outfile in [(train_ids, out_train), (val_ids, out_val)]:
        with open(outfile, 'w') as fp:
            for _id in split_ids:
                rec = df[df['id']==_id].iloc[0].to_dict()
                json.dump(rec, fp); fp.write('\n')

if __name__ == '__main__':
    preprocess_chexpert(
        csv_path='data/chexpert.csv',
        out_train='data/train.jsonl',
        out_val='data/valid.jsonl'
    )
