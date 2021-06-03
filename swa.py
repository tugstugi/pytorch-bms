"""
Stochastic Weight Averaging (SWA)

Averaging Weights Leads to Wider Optima and Better Generalization

https://github.com/timgaripov/swa
"""
from pathlib import Path
import warnings
import argparse

import torch
from tqdm import tqdm

from datasets.bms import BMSTrainDataset
from datasets.transforms import get_test_transform, Compose
from eval import bms_collate


def moving_average(net1, net2, alpha=1.):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, device):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param dataset: train dataset for buffers average estimation.
        :param model: model being update
        :param jobs: jobs for dataloader
        :return: None
    """
    if not check_bn(model):
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('no bn in model?!')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>!')
        # return model

    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0

    model = model.to(device)
    pbar = tqdm(loader, unit="samples", unit_scale=loader.batch_size)
    for sample in pbar:
        inputs, targets, target_lengths = sample['input'].to(device), sample['label'].to(device), sample['label_length'].to(device)

        inputs = inputs.to(device)
        b = inputs.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        # model(inputs)
        # TODO:
        model(inputs, False, targets, target_lengths, 275, test_dataset.tokenizer)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    return model


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    # project imports
    from models import get_model
    from misc.utils import load_checkpoint

    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help='input directory')
    parser.add_argument("--output", type=str, default='swa_model.pth', help='output model file')

    parser.add_argument("--dataload-workers-nums", type=int, default=4, help='number of workers for dataloader')

    parser.add_argument("--image-size", default=224, type=int, help="image size")
    parser.add_argument("--batch-size", type=int, default=64, help='valid batch size')

    parser.add_argument('--bn-update', action='store_true', help='update batch norm')
    args = parser.parse_args()

    test_transform = get_test_transform(args)
    test_dataset = BMSTrainDataset(fold=0, mode='train', transform=test_transform, dataset_size=0)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=bms_collate,
                                  num_workers=args.dataload_workers_nums,
                                  sampler=None, pin_memory=True, drop_last=True)

    if ',' in args.input:
        files = []
        for f in args.input.split(','):
            assert f.endswith('.pth')
            assert Path(f).exists()
            files.append(f)
    else:
        directory = Path(args.input)
        files = [f for f in directory.iterdir() if f.suffix == ".pth"]
    assert (len(files) > 1)

    model_args = torch.load(files[0], map_location='cpu').get('args', None)


    def load_model(f):
        model = get_model(model_args)
        load_checkpoint(f, model, optimizer=None, use_gpu=True, remove_module_keys=True)
        model.float()
        return model.cuda()


    def save_model(model, f):
        torch.save({
            'epoch': -1,
            'global_step': -1,
            'state_dict': model.state_dict(),
            'optimizer': {},
            'args': model_args
        }, f)


    net = load_model(files[0])
    for i, f in enumerate(files[1:]):
        net2 = load_model(f)
        moving_average(net, net2, 1. / (i + 2))

    if args.bn_update:
        with torch.no_grad():
            net = bn_update(test_data_loader, net, torch.device('cuda'))

    save_model(net, args.output)
