import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class ROIAwareGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, demographic_dim):
        super(ROIAwareGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

        self.fc1 = nn.Linear(out_channels + demographic_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # For binary classification

    def forward(self, data):
        x = torch.relu(self.conv1(data.x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        x = torch.relu(self.conv3(x, data.edge_index))
        x = global_mean_pool(x, data.batch)

        # Combine with demographic data
        x = torch.cat([x, data.demographic], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
