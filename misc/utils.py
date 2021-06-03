__author__ = 'Erdene-Ochir Tuguldur'

import os
import glob
import torch


def get_last_checkpoint_file_name(logdir):
    """Returns the last checkpoint file name in the given log dir path."""
    # checkpoints = glob.glob(os.path.join(logdir, '*.pth'))
    # checkpoints.sort()
    # if len(checkpoints) == 0:
    #     return None
    # return checkpoints[-1]
    checkpoint = os.path.join(logdir, 'last.pth')
    if os.path.exists(checkpoint):
        return checkpoint
    return None


def load_checkpoint(checkpoint_file_name, model, optimizer, use_gpu=False,
                    remove_module_keys=True, remove_decoder=False, remove_encoder=False, non_strict=False):
    """Loads the checkpoint into the given model and optimizer."""
    checkpoint = torch.load(checkpoint_file_name, map_location='cpu' if not use_gpu else None)
    state_dict = checkpoint['state_dict']
    if remove_module_keys or remove_decoder:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
            if remove_encoder and k.startswith('encoder'):
                del new_state_dict[k]
            if remove_decoder and k.startswith('decoder'):
                del new_state_dict[k]
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False if (remove_encoder or remove_decoder or non_strict) else True)
    model.float()
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    saved_args = checkpoint.get('args', None)
    del checkpoint
    print("loaded checkpoint epoch=%d step=%d" % (start_epoch, global_step))
    return start_epoch, global_step, saved_args


def save_checkpoint(logdir, epoch, global_step, model, optimizer, args, checkpoint_file_name):
    """Saves the training state into the given log dir path."""
    # checkpoint_file_name = os.path.join(logdir, 'epoch-%04d.pth' % epoch)
    # print("saving the checkpoint file '%s'..." % checkpoint_file_name)
    checkpoint = {
        'epoch': epoch + 1,
        'global_step': global_step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args
    }
    torch.save(checkpoint, checkpoint_file_name)
    del checkpoint


def save_latest_checkpoint(logdir, epoch, global_step, model, optimizer, args):
    # checkpoint_file_name = os.path.join(logdir, 'last.pth')
    checkpoint_file_name = os.path.join(logdir, 'epoch-%d.pth' % epoch)
    if os.path.exists(checkpoint_file_name):
        os.remove(checkpoint_file_name)
    save_checkpoint(logdir, epoch, global_step, model, optimizer, args, checkpoint_file_name)


def save_best_checkpoint(logdir, epoch, global_step, model, optimizer, args, metric, name):
    """Saves the training state into the given log dir path."""
    checkpoint_file_name = os.path.join(logdir, 'best-%s.pth' % name)
    # print("saving the checkpoint file '%s'..." % checkpoint_file_name)
    checkpoint = {
        'epoch': epoch + 1,
        'global_step': global_step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args,
        'metric': metric
    }
    torch.save(checkpoint, checkpoint_file_name)
    del checkpoint
