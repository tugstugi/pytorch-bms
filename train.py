#!/usr/bin/env python
"""Train for the Kaggle Bristol-Myers Squibb – Molecular Translation challenge: https://www.kaggle.com/c/bms-molecular-translation"""
__author__ = 'Erdene-Ochir Tuguldur'

import cv2
import os
import json
import time
import torch
import numpy as np
import random
import argparse

import warnings

from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist

from datasets.bms import BMSTrainDataset, BMSPseudoDataset, BMSExtraDataset
from misc.metrics import compute_metric

warnings.filterwarnings("ignore")

from tqdm import *

# seed
seed = 1234
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


import apex
from apex.parallel import DistributedDataParallel
from apex import amp
import albumentations as album

from torch.utils.data import DataLoader, Subset, ConcatDataset

from tensorboardX import SummaryWriter

# project imports
from datasets import *
from losses import *
from models import get_model
from optims import get_optimizer, get_lr_scheduler
from misc.utils import save_best_checkpoint, save_latest_checkpoint

from datasets.transforms import Compose, ApplyAlbumentations, get_test_transform, CropAugment, RandomNoiseAugment
from misc.utils import load_checkpoint

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--logdir", type=str, default=None, help='log dir for tensorboard logs and checkpoints')
parser.add_argument("--train-batch-size", type=int, default=64, help='train batch size')
parser.add_argument("--valid-batch-size", type=int, default=64, help='train batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=0, help='weight decay')
parser.add_argument("--optim", default='adam', help='choices of optimization algorithms')
parser.add_argument("--clip-grad-norm", type=float, default=0, help='clip gradient norm value')
parser.add_argument("--encoder-lr", type=float, default=1e-3, help='encoder learning rate for optimization')
parser.add_argument("--decoder-lr", type=float, default=1e-3, help='decoder learning rate for optimization')
parser.add_argument("--min-lr", type=float, default=1e-12, help='minimal learning rate for optimization')
parser.add_argument("--warm-start", type=str, help='warm start from a checkpoint')
parser.add_argument("--lr-warmup-steps", type=int, default=1000, help='learning rate warmup steps')
parser.add_argument("--lr-gamma", type=float, default=0.1, help='learning rate gamma for step scheduler')
parser.add_argument("--lr-step-ratio", type=lambda s: [float(item) for item in s.split(',')], default=[0.4],
                    help='learning rate step ratio for step scheduler')
parser.add_argument("--lr-policy", choices=['cosine', 'mcosine', 'step', 'mstep', 'none', 'plateau'], default='none',
                    help='learning rate scheduling policy')
parser.add_argument("--lr-plateau_mode", choices=['min', 'max'], default='min', help='reduce on plateau mode')
parser.add_argument("--lr-plateau_patience", type=int, default=10, help='reduce on plateau patience')
parser.add_argument('--mixed-precision', action='store_true', help='enable mixed precision training')
parser.add_argument('--amp-level', type=str, default='O2', help='amp level')
parser.add_argument('--sync-bn', action='store_true', help='enable apex sync batch norm.')
parser.add_argument('--cudnn-benchmark', action='store_true', help='enable CUDNN benchmark')
parser.add_argument("--max-epochs", default=20, type=int, help="train epochs")
parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument("--encoder-freeze", default=0, type=int, help="freeze first n encoder layers of encoder")
parser.add_argument("--decoder-freeze", default=0, type=int, help="freeze first n encoder layers of decoder")
parser.add_argument('--remove-decoder', action='store_true', help='remove decoder weights for warm start')
parser.add_argument('--remove-encoder', action='store_true', help='remove encoder weights for warm start')
parser.add_argument("--non-strict", action='store_true', help="non strict loading")

parser.add_argument('--debug', action='store_true', help='visual debug')
parser.add_argument('--cache', action='store_true', help='cache training data')
parser.add_argument('--verbose', action='store_true', help='cache training data')
parser.add_argument('--pipeline', action='store_true', help='make progress bar less verbose')

parser.add_argument("--valid-epochs", default=1, type=int, help='validation at every valid-epochs')
parser.add_argument('--reset-state', action='store_true', help='reset global steps and epochs to 0')

parser.add_argument("--model", default='swin_base_patch4_window12_384', help='choices of neural network')
parser.add_argument("--loss", default='ce', choices=LOSSES, help='choices of loss')
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--image-size", default=224, type=int, help="image size")
parser.add_argument("--train-dataset-size", default=0, type=int, help="subsample train set")
parser.add_argument("--valid-dataset-size", default=0, type=int, help="subsample validation set")
parser.add_argument("--max-token", default=275, type=int, help="max token")
parser.add_argument('--valid-dataset-non-sorted', action='store_true', help='reset global steps and epochs to 0')

parser.add_argument("--pseudo", default=None, type=str, help="pseudo dataset")
parser.add_argument("--pseudo-dataset-size", default=0, type=int, help="subsample pseudo dataset")

parser.add_argument("--extra", action='store_true', help="use extra images dataset")
parser.add_argument("--extra-dataset-size", default=0, type=int, help="subsample extra images dataset")

parser.add_argument("--embed-dim", default=384, type=int, help="embedding dim")
parser.add_argument("--vocab-size", default=193, type=int, help="vocab size")

parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing alpha")

parser.add_argument("--aug-rotate90-p", default=0.5, type=float, help="rotate probability")
parser.add_argument("--aug-crop-p", default=0.0, type=float, help="border crop probability")
parser.add_argument("--aug-noise-p", default=0.0, type=float, help="noise probability")

parser.add_argument("--num-head", default=8, type=int, help="decoder num head")
parser.add_argument("--num-layer", default=3, type=int, help="decoder num layer")
parser.add_argument("--ff-dim", default=1024, type=int, help="decoder ff dim")

args = parser.parse_args()

args.distributed = False
args.world_size = 1
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
torch.backends.cudnn.benchmark = args.cudnn_benchmark

train_transform = Compose([
    CropAugment(probability=args.aug_crop_p, crops=list(range(5, 20))),
    ApplyAlbumentations(album.Compose([
        # album.Resize(args.image_size, args.image_size),
        # album.HorizontalFlip(p=0.5),
        # album.VerticalFlip(p=0.5),
        album.RandomRotate90(p=args.aug_rotate90_p),

        # album.RandomScale(scale_limit=0.1, p=1),
        # album.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),
        # album.PadIfNeeded(args.image_size, args.image_size, border_mode=cv2.BORDER_CONSTANT, p=1),
        # album.RandomCrop(args.image_size, args.image_size, p=1),
        # album.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.7),

        # album.CoarseDropout(p=0.3),
        # album.GaussNoise(p=0.3),
        # album.IAASharpen(p=0.1)
    ])),
    RandomNoiseAugment(probability=args.aug_noise_p, frac=0.001),
    get_test_transform(args)
])
valid_transform = get_test_transform(args)

train_dataset = BMSTrainDataset(fold=args.fold, mode='train', transform=train_transform, dataset_size=args.train_dataset_size)
valid_dataset = BMSTrainDataset(fold=args.fold, mode='valid', transform=valid_transform, dataset_size=args.valid_dataset_size,
                                sort_valid=not args.valid_dataset_non_sorted)
tokenizer = train_dataset.tokenizer

pseudo_and_extra_datasets = []
if args.pseudo:
    pseudo_dataset = BMSPseudoDataset(pseudo_file=args.pseudo, transform=train_transform, dataset_size=args.pseudo_dataset_size)
    pseudo_and_extra_datasets.append(pseudo_dataset)

if args.extra:
    extra_dataset = BMSExtraDataset(transform=train_transform, dataset_size=args.extra_dataset_size)
    pseudo_and_extra_datasets.append(extra_dataset)

if len(pseudo_and_extra_datasets) > 0:
    train_dataset = ConcatDataset([train_dataset] + pseudo_and_extra_datasets)


def bms_collate(batch):
    inputs, labels, label_lengths, inchis = [], [], [], []
    for b in batch:
        inputs.append(b['input'])
        labels.append(torch.LongTensor(b['label']).reshape(-1, 1))
        label_lengths.append(torch.LongTensor([b['label_length']]))
        inchis.append(b['inchi'])
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    return {
        'input': torch.stack(inputs),
        'label': labels.squeeze(dim=-1),
        'label_length': torch.stack(label_lengths).reshape(-1, 1),
        'inchi': inchis
    }


train_data_sampler, valid_data_sampler = None, None
if args.distributed:
    train_data_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_data_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)
train_data_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(train_data_sampler is None),
                               collate_fn=bms_collate,
                               num_workers=args.dataload_workers_nums,
                               sampler=train_data_sampler, pin_memory=True, drop_last=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False,
                               collate_fn=bms_collate,
                               num_workers=args.dataload_workers_nums,
                               sampler=valid_data_sampler, pin_memory=True)

model = get_model(args)
if args.verbose and args.local_rank == 0:
    print(model)

if args.warm_start:
    load_checkpoint(args.warm_start, model, optimizer=None, use_gpu=False,
                    remove_encoder=args.remove_encoder, remove_decoder=args.remove_decoder, non_strict=args.non_strict)

if args.sync_bn:
    model = apex.parallel.convert_syncbn_model(model)
model = model.cuda()

criterion = get_loss(args, tokenizer)

if args.encoder_freeze != 0:
    idx = 0
    for idx, parameter in enumerate(model.encoder.parameters()):  # enumerate(model.encoder[:args.encoder_freeze].parameters()):
        parameter.requires_grad = False
    if args.local_rank == 0:
        print("encoder frozen!")
        # print("freezing %i n layers of total %i encoder layers" % (idx + 1, len(model.encoder)))
if args.decoder_freeze != 0:
    idx = 0
    for idx, parameter in enumerate(model.decoder.parameters()):  # enumerate(model.decoder[:args.decoder_freeze].parameters()):
        parameter.requires_grad = False
    if args.local_rank == 0:
        print("decoder frozen!")

optimizer = get_optimizer(args, model)

total_steps = int(len(train_dataset) * args.max_epochs / (args.world_size * args.train_batch_size))
if args.local_rank == 0:
    print("total steps:", total_steps, " epoch steps:", int(total_steps / args.max_epochs))

lr_scheduler = get_lr_scheduler(args, optimizer)

if args.mixed_precision:
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_level)  # , min_loss_scale=1)
if args.distributed:
    model = DistributedDataParallel(model)

start_timestamp = int(time.time() * 1000)
start_epoch = 1
global_step = 0
best_metric = 999
best_loss = 1e9

if args.logdir is None:
    logname = "%s_%s_wd%.0e" % (args.model, args.optim, args.weight_decay)
    if args.comment:
        logname = "%s_%s" % (logname, args.comment.replace(' ', '_'))
    logdir = os.path.join('logdir', logname)
else:
    logdir = args.logdir

writer = SummaryWriter(log_dir=logdir)
if args.local_rank == 0:
    print(vars(args))
    writer.add_text("hparams", json.dumps(vars(args), indent=4))


def get_lr():
    return optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']


def lr_warmup(step):
    if step < args.lr_warmup_steps:
        assert len(optimizer.param_groups) == 2

        encoder_lr = args.encoder_lr * (step + 1) / (args.lr_warmup_steps + 1)
        optimizer.param_groups[0]['lr'] = encoder_lr
        decoder_lr = args.decoder_lr * (step + 1) / (args.lr_warmup_steps + 1)
        optimizer.param_groups[1]['lr'] = decoder_lr
    elif step == args.lr_warmup_steps:
        optimizer.param_groups[0]['lr'] = args.encoder_lr
        optimizer.param_groups[1]['lr'] = args.decoder_lr


def train(epoch, phase='train'):
    global global_step, best_metric

    lr_warmup(global_step)
    if args.local_rank == 0:
        lrs = get_lr()
        print("epoch %3d with encoder_lr=%.02e decoder_lr=%.02e" % (epoch, lrs[0], lrs[1]))

    if args.distributed:
        train_data_sampler.set_epoch(epoch)

    model.train() if phase == 'train' else model.eval()
    torch.set_grad_enabled(True) if phase == 'train' else torch.set_grad_enabled(False)
    data_loader = train_data_loader if phase == 'train' else valid_data_loader

    it = 0
    running_loss = 0.0
    all_inchis = []
    all_predictions = []

    pbar = None
    if args.local_rank == 0:
        batch_size = args.train_batch_size if phase == 'train' else args.valid_batch_size
        pbar = tqdm(data_loader, unit="images", unit_scale=batch_size,
                    disable=False, mininterval=30.0 if args.pipeline else 0.1)

    for batch in data_loader if pbar is None else pbar:
        inputs, targets, target_lengths = batch['input'].cuda(), batch['label'].cuda(), batch['label_length'].cuda()

        if args.debug:
            print(batch['inchi'])
            print(inputs.size(), targets.size(), target_lengths.size())
            img = inputs[0, 0, :, :].detach().cpu().numpy()
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()

        _criterion = criterion

        loss = None
        if phase == 'train':
            targets, outputs = model(inputs, False, targets, target_lengths, args.max_token, tokenizer)
            loss = _criterion(outputs, targets).mean()
        else:
            predictions = model(inputs, True, targets, target_lengths, args.max_token, tokenizer)
            all_predictions += predictions
            all_inchis += batch['inchi']

        if phase == 'train':
            lr_warmup(global_step)
            optimizer.zero_grad()

            if args.mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.clip_grad_norm > 0:
                # clip gradient
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip_grad_norm)

            optimizer.step()

            # global step size is increased only in the train phase
            global_step += 1
        it += 1

        if loss is not None:
            loss = loss.item()
            running_loss += loss

        if args.local_rank == 0:
            if global_step % 5 == 1:
                if phase == 'train':
                    writer.add_scalar('%s/loss' % phase, loss, global_step)
                    lrs = get_lr()
                    writer.add_scalar('%s/encoder_lr' % phase, lrs[0], global_step)
                    writer.add_scalar('%s/decoder_lr' % phase, lrs[1], global_step)

                # update the progress bar
                pbar.set_postfix({
                    'loss': "%.05f" % (running_loss / it)
                })

            if not args.pipeline:
                # update the progress bar
                pbar.set_postfix({
                    'loss': "%.05f" % (running_loss / it)
                })

    epoch_loss = running_loss / it
    metric = 999
    if phase == 'valid':
        metric = compute_metric(all_inchis, all_predictions)

        if args.distributed:
            metric_tensor = torch.tensor(metric).cuda()
            rt = metric_tensor.clone()
            dist.all_reduce(rt, op=dist.reduce_op.SUM)
            rt = rt / args.world_size
            metric = rt.item()

    if args.local_rank == 0:
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)

        if phase == 'valid':
            writer.add_scalar('%s/metric' % phase, metric, epoch)
            print(all_inchis[0])
            print(all_predictions[0])
            print("Metric: %f" % metric)

            writer.add_text('%s/prediction' % phase,
                            'truth: %s\npredicted: %s' % (all_inchis[0], all_predictions[0]),
                            global_step if phase == 'train' else global_step + it)

            save_latest_checkpoint(logdir, epoch, global_step, model, optimizer, args)
            if metric <= best_metric:
                best_metric = metric
                print("\nBest metric: %.2f" % best_metric)
                save_best_checkpoint(logdir, epoch, global_step, model, optimizer, args, metric, 'metric')

        writer.flush()

    return epoch_loss


since = time.time()
epoch = start_epoch
valid_epoch_loss = 1e6
while True:
    train_epoch_loss = train(epoch, phase='train')

    if args.local_rank == 0:
        time_elapsed = time.time() - since
        time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600,
                                                                         time_elapsed % 3600 // 60,
                                                                         time_elapsed % 60)
        print("train epoch loss %f, step=%d, %s" % (train_epoch_loss, global_step, time_str))
    if epoch == args.max_epochs or epoch % args.valid_epochs == 0:
        valid_epoch_loss = train(epoch, phase='valid')
        if args.local_rank == 0:
            # print("valid epoch loss %f\n############\n" % valid_epoch_loss)
            print()
    if lr_scheduler is not None:
        if args.lr_policy == 'plateau':
            print("plateau")
            lr_scheduler.step(valid_epoch_loss)
        else:
            lr_scheduler.step()

    epoch += 1

    if epoch > args.max_epochs:
        break

writer.close()
