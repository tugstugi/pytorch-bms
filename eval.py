#!/usr/bin/env python

__author__ = 'Erdene-Ochir Tuguldur'

import os
import argparse

from torch.nn.utils.rnn import pad_sequence
from tqdm import *

# project imports
from datasets.bms import BMSTrainDataset, BMSTestDataset
from models import get_model

from datasets.transforms import *
from misc.metrics import *

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

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


def bms_collate(batch):
    inputs, labels, label_lengths, inchis = [], [], [], []
    for b in batch:
        inputs.append(b['input'])
        labels.append(torch.LongTensor(b['label']).reshape(-1, 1))
        label_lengths.append(torch.LongTensor([b['label_length']]))
        inchis.append(b['inchi'])
    labels = pad_sequence(labels, batch_first=True, padding_value=192)
    return {
        'input': torch.stack(inputs),
        'label': labels.squeeze(dim=-1),
        'label_length': torch.stack(label_lengths).reshape(-1, 1),
        'inchi': inchis
    }


def load_model_and_data(add_args, validation=True):
    checkpoint_file_name = add_args.checkpoint
    print(checkpoint_file_name)
    checkpoint = torch.load(checkpoint_file_name, map_location='cpu')
    args = checkpoint.get('args', None)
    print("saved args", args)
    if 'metric' in checkpoint:
        print("metric %.2f%%" % checkpoint['metric'])
    if not hasattr(args, 'max_token'):
        args.max_token = 275
    if hasattr(add_args, 'test_fold') and add_args.test_fold is not None:
        print("\n\n\n\n\n\nWARNING: this is evil don't use it...!!!\n\n\n\n\n\n", args)
        args.fold = add_args.test_fold

    model = get_model(args)
    state_dict = checkpoint['state_dict']
    remove_module_keys = True
    if remove_module_keys:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[len('module.'):]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.float()
    start_epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)

    del checkpoint
    print("loaded checkpoint epoch=%d step=%d" % (start_epoch, global_step))

    args.dataload_workers_nums = add_args.dataload_workers_nums

    test_transform = get_test_transform(args)

    if validation:
        test_dataset = BMSTrainDataset(fold=args.fold, mode='valid', transform=test_transform, dataset_size=0)
        test_batch_size = add_args.batch_size
        collate_fn = bms_collate
    else:
        test_transform = Compose([
            TestFix(),
            test_transform
        ])
        test_dataset = BMSTestDataset(transform=test_transform)
        test_batch_size = add_args.batch_size
        collate_fn = None

    test_data_sampler = None
    if add_args.distributed:
        test_data_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)

    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                                  collate_fn=collate_fn,
                                  num_workers=args.dataload_workers_nums,
                                  sampler=test_data_sampler, pin_memory=True)

    return model, args, test_dataset, test_batch_size, test_data_loader


def evaluate(add_args):
    model, args, test_dataset, test_batch_size, test_data_loader = load_model_and_data(add_args, validation=True)

    model.eval()
    model.cuda()
    torch.set_grad_enabled(False)
    if add_args.mixed_precision:
        model = model.half()

    all_inchis = []
    all_predictions = []

    print(test_dataset.tokenizer.stoi["<pad>"])

    pbar = tqdm(test_data_loader, unit="images", unit_scale=test_batch_size, mininterval=10)
    for batch in pbar:
        inputs = batch['input'].cuda()
        if add_args.mixed_precision:
            inputs = inputs.half()

        predictions = model(inputs, True, None, None, args.max_token, test_dataset.tokenizer)
        all_predictions += predictions
        all_inchis += batch['inchi']

    print(all_inchis[0])
    print(all_predictions[0])
    metric = compute_metric(all_inchis, all_predictions)
    print(metric)

    if add_args.distributed:
        metric_tensor = torch.tensor(metric).cuda()
        rt = metric_tensor.clone()
        dist.all_reduce(rt, op=dist.reduce_op.SUM)
        rt = rt / add_args.world_size
        metric = rt.item()

        # wait until everything finished
        dist.barrier()

    # import pickle
    # pickle.dump({
    #    'inchis': all_inchis,
    #    'predictions': all_predictions
    # }, open('eval.pickle', 'wb'))

    return metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataload-workers-nums", type=int, default=4, help='number of workers for dataloader')
    parser.add_argument("--test-fold", type=int, help="don't use it!!!")
    parser.add_argument("--batch-size", type=int, default=64, help='batch size')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--cudnn-benchmark', action='store_true', help='enable CUDNN benchmark')
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

    metric = evaluate(main_args)
    if main_args.local_rank == 0:
        print("%s: %.2f" % ('metric', metric))
