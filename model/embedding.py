import torch
import torch.nn as nn
import torch.nn.functional as F

from metric.utils import pdist


class L2NormEmbedding(nn.Module):
    def __init__(
        self, base, feature_size=512, embedding_size=128,
    ):
        super(L2NormEmbedding, self).__init__()
        self.base = base
        self.linear = nn.Linear(feature_size, embedding_size)

    def forward(self, x):
        feat = self.base(x)
        feat = feat.view(x.size(0), -1)
        embedding = self.linear(feat)
        embedding = F.normalize(embedding, dim=1, p=2)

        return embedding


class Embedding(nn.Module):
    def __init__(self, base, feature_size=512, embedding_size=128):
        super(Embedding, self).__init__()
        self.base = base
        self.linear = nn.Linear(feature_size, embedding_size)

    def forward(self, x):
        feat = self.base(x)
        feat = feat.view(x.size(0), -1)
        embedding = self.linear(feat)

        if self.training:
            # Please check "Learning without L2 Norm" in Section 3.1
            dist_mat = pdist(embedding)
            mean_d = dist_mat[
                ~torch.eye(dist_mat.shape[0], dtype=torch.bool, device=dist_mat.device)
            ].mean()
            return embedding / mean_d

        return embedding
