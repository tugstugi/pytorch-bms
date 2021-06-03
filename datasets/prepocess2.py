#
# Original code: https://www.kaggle.com/yasufuminakama/inchi-preprocess-2
#

import re
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


# ====================================================
# Preprocess functions
# ====================================================
def split_form(form):
    string = ''
    for i in re.findall(r"[A-Z][^A-Z]*", form):
        elem = re.match(r"\D+", i).group()
        num = i.replace(elem, "")
        if num == "":
            string += f"{elem} "
        else:
            string += f"{elem} {str(num)} "
    return string.rstrip(' ')


def split_form2(form):
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'\d+', j))
            assert len(num_list) == 1, f"len(num_list) != 1"
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')


if __name__ == '__main__':
    # ====================================================
    # Data Loading
    # ====================================================
    data_root = '/data/bms/'
    train = pd.read_csv(data_root + 'train_labels.csv')
    print(f'train.shape: {train.shape}')

    train['InChI_1'] = train['InChI'].progress_apply(lambda x: x.split('/')[1])
    train['InChI_text'] = train['InChI_1'].progress_apply(split_form) + ' ' + \
                          train['InChI'].apply(lambda x: '/'.join(x.split('/')[2:])).progress_apply(split_form2).values
    # ====================================================
    # create tokenizer
    # ====================================================
    from tokenizer2 import Tokenizer
    import pickle

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train['InChI_text'].values)
    pickle.dump({
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos
    }, open(data_root + 'tokenizer2.pickle', 'wb'))
    print('Saved tokenizer2.pickle')
    # ====================================================
    # preprocess train.csv
    # ====================================================
    lengths = []
    tk0 = tqdm(train['InChI_text'].values, total=len(train))
    for text in tk0:
        seq = tokenizer.text_to_sequence(text)
        length = len(seq) - 2
        lengths.append(length)
    train['InChI_length'] = lengths
    train.to_pickle(data_root + 'train2.pkl')
    print('Saved preprocessed train2.pkl')
