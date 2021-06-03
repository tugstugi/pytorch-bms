__author__ = 'Erdene-Ochir Tuguldur'

import torch
import random
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

from .tokenizer2 import load_tokenizer

RANDOM_SEED = 1234


class BMSTrainDataset(Dataset):

    def __init__(self, fold=0, mode='train', data_root='/data/bms', transform=None, dataset_size=0, sort_valid=True):
        self.mode = mode
        self.data_root = Path(data_root)
        self.transform = transform
        self.dataset_size = dataset_size
        self.tokenizer = load_tokenizer(data_root)

        data = pd.read_pickle(str(self.data_root / 'train2.pkl'))

        # data = data[:500000]

        def gen_file_path(image_id):
            return str(self.data_root / "train/{}/{}/{}/{}.png".format(image_id[0], image_id[1], image_id[2], image_id))

        data['file_path'] = data['image_id'].apply(gen_file_path)

        pd.set_option('display.max_colwidth', None)
        # print(data['file_path'].head())
        # print(data['InChI_length'].max())

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        for n, (train_index, val_index) in enumerate(skf.split(data, data['InChI_length'])):
            data.loc[val_index, 'fold'] = int(n)
        data['fold'] = data['fold'].astype(int)
        # print(data.groupby(['fold']).size())

        if self.mode == 'train':
            data_idx = data[data['fold'] != fold].index
        else:
            data_idx = data[data['fold'] == fold].index

        self.df = data.loc[data_idx].reset_index(drop=True)
        if self.mode != 'train' and sort_valid:
            # make fast eval by sorting length
            self.df = self.df.sort_values(by=['InChI_length'])
        # print(self.df.head())
        # print(len(self.df))

    def __len__(self):
        if self.dataset_size > 0:
            return self.dataset_size
        return len(self.df)

    def __getitem__(self, idx):
        if self.dataset_size > 0 and self.mode == 'train':
            idx = random.randint(0, len(self.df) - 1)

        row = self.df.iloc[idx]

        image = Image.open(row['file_path']).convert('RGB')
        label = self.tokenizer.text_to_sequence(row['InChI_text'])

        data = {
            'input': np.array(image),
            'label': np.array(label),
            'label_length': len(label),
            'inchi': str(row['InChI'])
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


class BMSPseudoDataset(BMSTrainDataset):

    def __init__(self, pseudo_file, data_root='/data/bms', transform=None, dataset_size=0):
        self.mode = 'train'
        self.data_root = Path(data_root)
        self.transform = transform
        self.dataset_size = dataset_size
        self.tokenizer = load_tokenizer(data_root)

        data = pd.read_pickle(str(self.data_root / pseudo_file))

        def gen_file_path(image_id):
            return str(self.data_root / "test/{}/{}/{}/{}.png".format(image_id[0], image_id[1], image_id[2], image_id))

        data['file_path'] = data['image_id'].apply(gen_file_path)

        pd.set_option('display.max_colwidth', None)
        print(data['file_path'].head())
        print(data['InChI_length'].max())

        self.df = data


class BMSExtraDataset(BMSTrainDataset):

    def __init__(self, data_root='/data/bms', transform=None, dataset_size=0):
        self.mode = 'train'
        self.data_root = Path(data_root)
        self.transform = transform
        self.dataset_size = dataset_size
        self.tokenizer = load_tokenizer(data_root)

        data = pd.read_pickle(str(self.data_root / 'extra.pkl'))

        def gen_file_path(image_id):
            return str(self.data_root / "extra_images/{}/{}/{}/{}.png".format(image_id[0], image_id[1], image_id[2], image_id))

        data['file_path'] = data['image_id'].apply(gen_file_path)

        self.df = data
        self.df = self.df.sort_values(by=['InChI_length'], ascending=False)
        # self.df = self.df[:500000]

        pd.set_option('display.max_colwidth', None)
        print(self.df['file_path'].head())
        print(self.df['InChI_length'].min(), self.df['InChI_length'].mean(), self.df['InChI_length'].max())


class BMSTestDataset(Dataset):

    def __init__(self, data_root='/data/bms', transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.tokenizer = load_tokenizer(data_root)

        data = pd.read_csv(str(self.data_root / '4174.csv'))

        # data = data.sort_values(by=['InChI_length'])
        # data = data.drop(columns=['InChI_length'])

        def gen_file_path(image_id):
            return str(self.data_root / "test/{}/{}/{}/{}.png".format(image_id[0], image_id[1], image_id[2], image_id))

        data['file_path'] = data['image_id'].apply(gen_file_path)
        # data = data[:1024]
        self.df = data

        pd.set_option('display.max_colwidth', None)
        # print(data['file_path'].head())
        # print(self.df.head())
        # print(len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row['file_path']).convert('RGB')

        data = {
            'input': np.array(image),
            'image_id': row['image_id']
        }

        if self.transform is not None:
            data = self.transform(data)

        return data
