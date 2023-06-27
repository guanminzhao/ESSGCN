from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalization(data):

    for i in range(len(data)):

        _range = torch.max(data[i]) - torch.min(data[i])
        data[i] = (data[i] - torch.min(data[i])) / _range
    return data


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)

        # get batch_size
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)     # 16*1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)      # 16*16
        else:
            mask = mask.float().to(device)

        features = F.normalize(features, dim=2)
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
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)


        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()


        return loss




class  Syntactic_Reliability_ConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(Syntactic_Reliability_ConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, anchor_feature, label_feature, labels):
        """
        Args:
            anchor_feature: hidden vector of shape [bsz,1, hidden_size].
            label_feature: ground truth of shape [2,hidden_size].
            labels: contrastive mask of shape [bsz, 1],
        Returns:
            A loss scalar.
        """
        anchor_feature = F.normalize(anchor_feature, dim=2)
        contrast_feature = F.normalize(label_feature, dim=1)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature).squeeze(dim=1)

        # for numerical stability
        logits_max = torch.max(anchor_dot_contrast)
        logits = anchor_dot_contrast - logits_max.detach()


        # tile mask
        pos_label = (1 - labels).T
        neg_label = labels.T
        mask = torch.cat((pos_label, neg_label), dim=1)


        exp_logits = torch.exp(logits)


        log_prob = logits - (torch.log(exp_logits.sum(dim=1))).unsqueeze(dim=1).repeat(1,2)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum() / mask.sum()

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos


        return loss

if __name__ == '__main__':
    loss = Syntactic_Reliability_ConLoss()
    anchor = torch.rand([16, 1, 768]).cuda()
    labels_feature = torch.rand([2, 768]).cuda()
    labels = torch.Tensor([[0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]]).cuda()
    a = loss(anchor, labels_feature, labels)
    print(a)