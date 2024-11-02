import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=1, normalize=True):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.normalize = normalize

    def NTXent_Loss(self, features):
        # features: [2N, D], every positive sample pair is adjacent, that is, 2k-1 is the positive pair of 2k
        # 实际上由于最终是使用交叉熵实现的，最终只考虑了NTXent_Loss的分子部分，即是最大化样本与正样本的相似度，并没有显式地要求最小化样本与其他负样本的相似度

        features = F.normalize(features, dim=1) if self.normalize else features
        similarity_matrix = torch.matmul(features, features.T)  # Cosine similarity, [2N, 2N]
        similarity_matrix[torch.eye(features.size(0)).bool()] = float("-inf")

        target = torch.arange(8, device=similarity_matrix.device)
        target[0::2] += 1
        target[1::2] -= 1

        ''' Ground truth labels, 2k-1 is the positive pair of 2k
        tensor([[0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0]])
        '''

        # Standard cross entropy loss
        return F.cross_entropy(similarity_matrix / self.temperature, target, reduction="mean")


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR.
    source: from https://github.com/HobbitLong/SupContrast/tree/master
    """

    def __init__(self, temperature=1.0, contrast_mode='all', base_temperature=1.0, normalize=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.normalize = normalize

    def forward(self, features, labels=None, mask=None):
        # features: [N, V, D], V=0为anchor, v=1,2,...,V可以认为是根据不同augmentation得到的不同的view，和NTXent_Loss的输入不同
        # 最终的计算由mask决定，而labels只是用于计算mask
        # 如果没有augmentation，那么输入前将features.unsqueeze(1)即可，如果有augmentation，指定labels时，相当于还要求最大化所有增广的正样本的相似度，类似于NTXent_Loss
        # 实际上经过测试，最终只考虑了NTXent_Loss的分子部分，即是最大化样本与正样本的相似度，并没有显式地要求最小化样本与其他负样本的相似度，等同于CrossEntropy

        """Compute loss for loss. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = F.normalize(features, dim=2) if self.normalize else features
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class SimCSELoss(nn.Module):
    def __init__(self, temperature=1.0, mode='adjacent', normalize=True, reduction='mean'):
        super(SimCSELoss, self).__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.mode = mode  # 'adjacent' or 'separate'
        self.reduction = reduction  # 'none', 'mean' or 'sum', 'mean' takes the average of the cross-entropy loss over all samples

        # 'adjacent' mode: adjacent samples are positive pairs, e.g. [0, 1], [2, 3], [4, 5], ...
        # 'separate' mode: positive pairs are separated into two groups, e.g. [0, 1, 2, 3, 4, 5] -> [0, 3], [1, 4], [2, 5]

    def forward(self, features):
        if features.shape[0] % 2 != 0:
            raise ValueError("The number of features must be even.")

        features = F.normalize(features, dim=1) if self.normalize else features
        if self.mode == 'adjacent':
            ids = torch.arange(0, features.shape[0], device=features.device)
            labels = ids + 1 - ids % 2 * 2
            similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
            similarity_matrix = similarity_matrix - torch.eye(features.shape[0], device=features.device) * 1e12
            loss = F.cross_entropy(similarity_matrix / self.temperature, labels, reduction=self.reduction)  # sample-wise mean by default
        elif self.mode == 'separate':
            z1, z2 = torch.split(features, features.shape[0] // 2, dim=0)  # (2N, D) -> (N, D), (N, D)
            labels = torch.arange(0, z1.shape[0], device=features.device)  # (N,), [0, 1, 2, ..., N-1]
            similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2)  # (N, N)
            loss = F.cross_entropy(similarity_matrix / self.temperature, labels, reduction=self.reduction)  # sample-wise mean by default
        else:
            raise NotImplementedError
        return loss


class ContrastiveRankLoss(nn.Module):
    def __init__(self, compute_inverse=True):
        super(ContrastiveRankLoss, self).__init__()
        self.compute_inverse = compute_inverse

    def forward(self, logits, targets):
        # logits: [N, 1], N is the number of samples, 1 is the logit value
        # targets: [N, 1], N is the number of samples, 1 is the target value
        logits, targets = logits.reshape(-1, 1), targets.reshape(-1, 1)
        contrast_labels = targets - targets.transpose(1, 0)  # [N, N]
        contrast_labels = torch.sign(contrast_labels) * 0.5 + 0.5  # [N, N], range: (-inf, inf) -> {-1, 1} -> {0, 1}
        value_pred_diff = logits - logits.transpose(1, 0)  # [N, N]
        contrastive_logits = F.logsigmoid(value_pred_diff)  # [N, N]

        if self.compute_inverse:
            inverse_contrastive_logits = F.logsigmoid(-1 * value_pred_diff)
            losses = -contrast_labels * contrastive_logits - (1 - contrast_labels) * inverse_contrastive_logits
        else:
            losses = -contrast_labels * contrastive_logits

        mask = 1 - torch.eye(losses.shape[0], device=losses.device)  # [N, N], diagonal is 0 and others are 1
        contrastive_loss = torch.sum(losses * mask) / torch.sum(mask)  # take the average
        return contrastive_loss
