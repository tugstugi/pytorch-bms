#!/usr/bin/env python

__author__ = 'Erdene-Ochir Tuguldur'

import os
import glob
import numpy as np
import random
import argparse
from pathlib import Path
from tqdm import *

import torch
import torch.distributed as dist

# project imports
from eval import load_model_and_data

import warnings

warnings.filterwarnings("ignore")


seed = 1234
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def inference(add_args):
    model, args, test_dataset, test_batch_size, test_data_loader = load_model_and_data(add_args, validation=False)

    model.eval()
    model.cuda()
    torch.set_grad_enabled(False)
    if add_args.mixed_precision:
        model = model.half()

    print(test_dataset.tokenizer.stoi["<pad>"])

    with Path("%d.csv" % add_args.local_rank).open('wt') as f:
        # f.write('image_id,InChI\n')
        pbar = tqdm(test_data_loader, unit="images", unit_scale=test_batch_size, mininterval=30)
        for batch in pbar:
            inputs, image_ids = batch['input'].cuda(), batch['image_id']
            if add_args.mixed_precision:
                inputs = inputs.half()

            predictions = model(inputs, True, None, None, args.max_token, test_dataset.tokenizer)
            for image_id, p in zip(image_ids, predictions):
                f.write("%s,\"%s\"\n" % (image_id, p))
            f.flush()

    if add_args.distributed:
        # wait until everything finished
        dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataload-workers-nums", type=int, default=4, help='number of workers for dataloader')
    parser.add_argument("--batch-size", type=int, default=64, help='batch size')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--cudnn-benchmark', action='store_true', help='enable CUDNN benchmark')
    parser.add_argument("--submission", type=str, default='submission.csv', help="submission output")
    parser.add_argument('--mixed-precision', action='store_true', help='mixed precision')
    parser.add_argument("checkpoint", type=str, help='a pretrained neural network model')
    main_args = parser.parse_args()

    main_args.distributed = False
    main_args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        main_args.distributed = int(os.environ['WORLD_SIZE']) > 1
        main_args.world_size = int(os.environ['WORLD_SIZE'])
    if main_args.distributed:
        torch.cuda.set_device(main_args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.benchmark = main_args.cudnn_benchmark

    inference(main_args)

    if main_args.local_rank == 0:
        image_ids = set()
        submission_file = Path(main_args.submission)
        submission_file.parent.mkdir(exist_ok=True)
        with submission_file.open('wt') as f:
            f.write('image_id,InChI\n')
            output_files = glob.glob("[0-9].csv")
            for output_file in sorted(output_files):
                print("merging %s..." % output_file)
                with Path(output_file).open('rt') as of:
                    for line in of:
                        image_id = line.split(',')[0]
                        if image_id not in image_ids:
                            image_ids.add(image_id)
                            f.write(line)
        print("image ids: ", len(image_ids))
