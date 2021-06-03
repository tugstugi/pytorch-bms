__author__ = 'Erdene-Ochir Tuguldur'

from .cait import Cait
from .swin import Swin
from .vit import Vit


def get_model(args):
    name = args.model

    if name.lower().startswith('vit'):
        return Vit(name, args)
    elif name.lower().startswith('swin'):
        return Swin(name, args)
    elif name.lower().startswith('cait'):
        return Cait(name, args)
    else:
        raise RuntimeError("Unknown model! %s" % name)
