__author__ = 'Erdene-Ochir Tuguldur'

import torch

from losses.ls_loss import LabelSmoothingLoss, LabelSmoothingCrossEntropy

LOSSES = ['ce']


def get_loss(args, tokenizer):
    if args.loss == 'ce':
        if args.label_smoothing > 0:
            # return LabelSmoothingLoss(args.label_smoothing, tgt_vocab_size=len(tokenizer), ignore_index=tokenizer.stoi["<pad>"]).cuda()
            return LabelSmoothingCrossEntropy(args.label_smoothing, ignore_index=tokenizer.stoi["<pad>"]).cuda()
        else:
            return torch.nn.CrossEntropyLoss(ignore_index=tokenizer.stoi["<pad>"])
    else:
        raise RuntimeError("Unknown loss!")
