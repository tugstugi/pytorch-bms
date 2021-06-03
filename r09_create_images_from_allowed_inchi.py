# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Draw
import albumentations as A


def sp_noise(image):
    # https://gist.github.com/lucaswiman/1e877a164a69f78694f845eab45c381a
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(image.shape[:2])
    image[probs < .0002] = black
    image[probs > .9] = white
    return image


def noisy_inchi(inchi, inchi_path, add_noise=True, crop_and_pad=True):
    mol = Chem.MolFromInchi(inchi)
    d = Draw.rdMolDraw2D.MolDraw2DCairo(640, 640)
    # https://www.kaggle.com/stainsby/improved-synthetic-data-for-bms-competition-v3
    d.drawOptions().useBWAtomPalette()
    d.drawOptions().bondLineWidth = 1
    d.drawOptions().additionalAtomLabelPadding = np.random.uniform(0, .2)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    d.WriteDrawingText(inchi_path)
    if crop_and_pad:
        img = cv2.imread(inchi_path, cv2.IMREAD_GRAYSCALE)
        crop_rows = img[~np.all(img == 255, axis=1), :]
        img = crop_rows[:, ~np.all(crop_rows == 255, axis=0)]
        img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.imread(inchi_path, cv2.IMREAD_GRAYSCALE)
    if add_noise:
        img = sp_noise(img)
        cv2.imwrite(inchi_path, img)
    return img


def create_dataset(input_csv, out_dir):
    fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])

    s = pd.read_csv(input_csv)
    print(s)
    print(s['image_id'].values)

    # Create folders
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    unique_letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    for u1 in unique_letters:
        if not os.path.isdir(out_dir + u1 + '/'):
            os.mkdir(out_dir + u1 + '/')
        for u2 in unique_letters:
            if not os.path.isdir(out_dir + u1 + '/' + u2 + '/'):
                os.mkdir(out_dir + u1 + '/' + u2 + '/')
            for u3 in unique_letters:
                if not os.path.isdir(out_dir + u1 + '/' + u2 + '/' + u3 + '/'):
                    os.mkdir(out_dir + u1 + '/' + u2 + '/' + u3 + '/')

    image_ids = s['image_id'].values
    inchis = s['InChI'].values
    pbar = tqdm(range(len(image_ids)))
    full_size = 0
    for i in pbar:
        image_id = image_ids[i]
        inchi = inchis[i]
        out_file = out_dir + '{}/{}/{}/{}.png'.format(image_id[0], image_id[1], image_id[2], image_id)
        if not os.path.isfile(out_file):
            img2 = noisy_inchi(inchi, out_file)
        full_size += os.path.getsize(out_file)
        # print(img2.shape, out_file)
        # show_image(img2)
        pbar.set_postfix({'shape': img2.shape, 'size': full_size / (1024 * 1024)})
        if 0:
            h, w = img2.shape
            if h > w:
                img2 = fix_transform(image=img2)['image']
            cv2.imwrite(out_file, img2)


def add_image_ids(s, out_csv):
    index = s.index.values
    image_ids = []
    for ind in index:
        m = ind + 0x1000000000000
        val = hex(m).split('x')[-1][::-1]
        print(ind, val)
        image_ids.append(val)
    s['image_id'] = image_ids
    s.to_csv(out_csv, index=False)


if __name__ == '__main__':
    input_csv = '/data/bms/extra_approved_InChIs.csv'
    input_csv_fixed = '/data/bms/extra_approved_InChIs_with_ids.csv'
    s = pd.read_csv(input_csv)


    def compute_length(col):
        def _compute_length(row):
            return len(row[col])

        return _compute_length


    s['length'] = s.apply(compute_length('InChI'), axis=1)
    s = s.sort_values(by=['length'], ascending=False)

    # only longest 1m images
    s = s[:1000000]

    if not os.path.isfile(input_csv_fixed):
        add_image_ids(s, input_csv_fixed)
    output_dir = '/data/bms/extra_images/'
    create_dataset(input_csv_fixed, output_dir)
    # test_random_molecule_image(n=20, graphics=True)
