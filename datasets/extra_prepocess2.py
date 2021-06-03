__author__ = 'Erdene-Ochir Tuguldur'

import pandas as pd
from tqdm.auto import tqdm

from tokenizer2 import load_tokenizer
from prepocess2 import split_form, split_form2

tqdm.pandas()

if __name__ == '__main__':
    # ====================================================
    # Data Loading
    # ====================================================
    # cat PSD_43.csv | grep -v "?" | grep -v "p" | grep -v "q" > 43.csv
    train = pd.read_csv('/data/bms/extra_approved_InChIs_with_ids.csv')
    print(f'train.shape: {train.shape}')

    train['InChI_1'] = train['InChI'].progress_apply(lambda x: x.split('/')[1])
    train['InChI_text'] = train['InChI_1'].progress_apply(split_form) + ' ' + \
                          train['InChI'].apply(lambda x: '/'.join(x.split('/')[2:])).progress_apply(split_form2).values

    tokenizer = load_tokenizer()

    # ====================================================
    # preprocess pseudo
    # ====================================================
    lengths = []
    tk0 = tqdm(train['InChI_text'].values, total=len(train))
    for text in tk0:
        try:
            seq = tokenizer.text_to_sequence(text)
            length = len(seq) - 2
            lengths.append(length)
        except KeyError:
            lengths.append(-1)
    train['InChI_length'] = lengths
    train = train[train['InChI_length'] != -1]
    print("valid inchis: ", len(train))
    train.to_pickle('/data/bms/extra.pkl')
    print('Saved preprocessed pseudo.pkl')
