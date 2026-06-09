import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS


@HEADS.register_module()
class UncertaintyEstimator_w_feature_au(nn.Module):
    """Estimate per-voxel uncertainty of the spatiotemporal memory.

    Input per voxel: [observation count, distance, prev class distribution
    (18), prev voxel feature (feature_dim), current voxel feature
    (feature_dim), aleatoric uncertainty (1)].
    """

    def __init__(self, input_size=20, hidden_size=64, feature_dim=80, latent_dim=18):
        super(UncertaintyEstimator_w_feature_au, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)  # Final output layer: 1 uncertainty value per voxel

        self.prev_voxel_mapper = nn.Linear(feature_dim, latent_dim)
        self.voxel_query_mapper = nn.Linear(feature_dim, latent_dim)

        self.feature_dim = feature_dim

    def forward(self, x):
        _x = x[..., :20]
        prev_voxel = x[..., 20:20 + self.feature_dim]
        voxel_query = x[..., 20 + self.feature_dim: 20 + 2 * self.feature_dim]
        au = 1 - x[..., 20 + 2 * self.feature_dim:]

        _prev_voxel = self.prev_voxel_mapper(prev_voxel)
        _voxel_query = self.voxel_query_mapper(voxel_query)
        sim = nn.functional.cosine_similarity(_prev_voxel, _voxel_query, dim=-1).unsqueeze(1)

        x = torch.cat([_x, sim, au], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        uncertainty = torch.sigmoid(self.fc4(x))  # Output uncertainty in range [0, 1]

        return uncertainty
