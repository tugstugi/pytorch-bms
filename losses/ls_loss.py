import torch
import torch.nn as nn
import torch.nn.functional as F


#
# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/loss.py
#
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


#
# https://github.com/fastai/fastai/blob/8013797e05f0ae0d771d60ecf7cf524da591503c/fastai/layers.py#L300
#
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.1, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.eps, self.reduction, self.ignore_index = eps, reduction, ignore_index
        print("label smoothing: ", self.eps, self.reduction, self.ignore_index)

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':  loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)
