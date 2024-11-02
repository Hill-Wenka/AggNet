import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..datastructure.json_utils import json2list


class FocalLoss(nn.Module):
    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        '''
        The initialization of focal loss
        :param num_class: the number of class label
        :param alpha: the scalar factor for this criterion (1D Tensor, Variable)
        :param gamma: gamma > 0; reduces the relative loss for well-classified notebook (p > .5),
                        putting more focus on hard, misclassified notebook gamma(float, double)
        :param average: By default, the losses are averaged over observations for each minibatch.
                        However, if the field size_average is set to False, the losses are
                        instead summed for each minibatch.
        '''
        super(FocalLoss, self).__init__()

        alpha = torch.ones(num_class, 1) if alpha is None else torch.tensor(json2list(alpha))
        self.alpha = Variable(alpha)
        self.gamma = gamma
        self.num_class = num_class
        self.reduction = reduction

    def forward(self, logits, targets, sample_weights=None):
        N, C = logits.size(0), logits.size(1)
        P = F.softmax(logits, dim=1)  # [N, C]
        class_mask = Variable(logits.data.new(N, C).fill_(0))  # [N, C] all zero
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  # digital label -> one-hot encoding, [0, 1, 1] -> [[1, 0], [0, 1], [0, 1]]
        alpha = self.alpha[ids.data.view(-1)].view(-1, 1).to(logits.device)  # [N, 1] the class weight for each sample
        probs = (P * class_mask).sum(1).view(-1, 1)  # [N, 1] the probability of the true class
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * probs.log()  # [N, 1] the focal loss of each sample
        if sample_weights is not None:
            sample_weights = Variable(sample_weights).reshape(-1, 1).to(batch_loss.device)  # [N, 1] the weight of each sample
            batch_loss = batch_loss * sample_weights  # [N, 1] the weighted focal loss of each sample

        if self.reduction == 'none':
            loss = batch_loss
        elif self.reduction == 'mean':
            loss = batch_loss.mean()
        elif self.reduction == 'sum':
            loss = batch_loss.sum()
        else:
            raise NotImplementedError
        return loss


class WeightedLoss(nn.Module):
    def __init__(self, loss_type, reduction='mean', **kwargs):
        super(WeightedLoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        if self.loss_type == 'CrossEntropy':
            self.loss = nn.CrossEntropyLoss(reduction='none', **kwargs)
        elif self.loss_type == 'BCELoss':
            self.loss = nn.BCELoss(reduction='none', **kwargs)
        elif self.loss_type == 'BCEWithLogitsLoss':
            self.loss = nn.BCEWithLogitsLoss(reduction='none', **kwargs)
        elif self.loss_type == 'FocalLoss':
            self.loss = FocalLoss(reduction='none', **kwargs)
        else:
            raise NotImplementedError

    def forward(self, logits, target, sample_weight=None):
        loss = self.loss(logits, target)
        if sample_weight is not None:
            assert loss.shape == sample_weight.shape
            loss = loss * sample_weight

        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise NotImplementedError
        return loss


class MutualInformationLoss(nn.Module):
    def __init__(self, alpha):
        super(MutualInformationLoss, self).__init__()
        self.alpha = alpha

    def entropy(self, probs):
        return - (probs.mean(0) * torch.log(probs.mean(0) + 1e-12)).sum(0)

    def cond_entropy(self, probs):
        return - (probs * torch.log(probs + 1e-12)).sum(1).mean(0)

    def forward(self, logtis):
        # logtis: [N, C]
        probs = F.softmax(logtis, dim=-1)
        return -(self.entropy(probs) - self.alpha * self.cond_entropy(probs))
