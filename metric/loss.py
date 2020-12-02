import torch
import torch.nn as nn
import torch.nn.functional as F

from metric.utils import pdist


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.margin = margin

        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p == 2))

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        loss = F.triplet_margin_loss(
            anchor_embed,
            positive_embed,
            negative_embed,
            margin=self.margin,
            reduction="none",
        )

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class MDRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.levels = nn.Parameter(torch.tensor([-3.0, 0.0, 3.0]))
        self.momentum = 0.9

        momented_mean = torch.zeros(1)
        momented_std = torch.zeros(1)
        self.register_buffer("momented_mean", momented_mean)
        self.register_buffer("momented_std", momented_std)

        # The variable is used to check whether momented_mean and momented_std are initialized
        self.init = False

    def initialize_statistics(self, mean, std):
        self.momented_mean = mean
        self.momented_std = std
        self.init = True

    def forward(self, embeddings):
        dist_mat = pdist(embeddings)
        pdist_mat = dist_mat[
            ~torch.eye(dist_mat.shape[0], dtype=torch.bool, device=dist_mat.device,)
        ]
        dist_mat = dist_mat.view(-1)

        mean = dist_mat.mean().detach()
        std = dist_mat.std().detach()

        if not self.init:
            self.initialize_statistics(mean, std)
        else:
            self.momented_mean = (
                1 - self.momentum
            ) * mean + self.momentum * self.momented_mean
            self.momented_std = (
                1 - self.momentum
            ) * std + self.momentum * self.momented_std

        normalized_dist = (pdist_mat - self.momented_mean) / self.momented_std
        difference = (normalized_dist[None] - self.levels[:, None]).abs().min(dim=0)[0]
        loss = difference.mean()

        return loss
