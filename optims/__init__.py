__author__ = 'Erdene-Ochir Tuguldur'

import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch_optimizer import RAdam, Lookahead


def get_optimizer(args, model):
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, model.encoder.parameters()), 'lr': args.encoder_lr},
                                     {'params': filter(lambda p: p.requires_grad, model.decoder.parameters()), 'lr': args.decoder_lr}],
                                    lr=args.decoder_lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW([{'params': filter(lambda p: p.requires_grad, model.encoder.parameters()), 'lr': args.encoder_lr},
                                       {'params': filter(lambda p: p.requires_grad, model.decoder.parameters()), 'lr': args.decoder_lr}],
                                      lr=args.decoder_lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.encoder.parameters()), 'lr': args.encoder_lr},
                                      {'params': filter(lambda p: p.requires_grad, model.decoder.parameters()), 'lr': args.decoder_lr}],
                                     lr=args.decoder_lr, weight_decay=args.weight_decay,
                                     # betas=(args.beta1, args.beta2)
                                     )
    elif args.optim == 'lookahead_radam':
        optimizer = Lookahead(RAdam([{'params': filter(lambda p: p.requires_grad, model.encoder.parameters()), 'lr': args.encoder_lr},
                                     {'params': filter(lambda p: p.requires_grad, model.decoder.parameters()), 'lr': args.decoder_lr}],
                                    lr=args.decoder_lr, weight_decay=args.weight_decay),
                              alpha=0.5, k=5)
    else:
        raise RuntimeError("Unknown optimizer!")
    return optimizer


def get_lr_scheduler(args, optimizer):
    if args.lr_policy == 'step':
        step_size = round(args.max_epochs * args.lr_step_ratio[0])
        print("step size", step_size)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=args.lr_gamma)
    elif args.lr_policy == 'mstep':
        milestones = [round(args.max_epochs * step_ratio) for step_ratio in args.lr_step_ratio]
        print("milestones", milestones)
        scheduler = MultiStepLR(optimizer, milestones, gamma=args.lr_gamma)
    elif args.lr_policy == 'cosine':
        t_max = round(args.max_epochs * args.lr_step_ratio[0])  # / 2)
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.min_lr)
    elif args.lr_policy == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, args.lr_plateau_mode, args.lr_gamma, args.lr_plateau_patience, threshold=0.001)
    else:
        scheduler = None
    return scheduler
