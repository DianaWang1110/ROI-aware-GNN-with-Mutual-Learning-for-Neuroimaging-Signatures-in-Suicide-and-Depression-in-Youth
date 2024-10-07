import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class MutualLearningGCN(nn.Module):
    def __init__(self, in_channels_desikan, in_channels_destrieux, in_channels_fuzzy, hidden_channels, out_channels, demographic_dim):
        super(MutualLearningGCN, self).__init__()
        # GCN layers for each parcellation
        self.conv1_desikan = GCNConv(in_channels_desikan, hidden_channels)
        self.conv2_desikan = GCNConv(hidden_channels, out_channels)

        self.conv1_destrieux = GCNConv(in_channels_destrieux, hidden_channels)
        self.conv2_destrieux = GCNConv(hidden_channels, out_channels)

        self.conv1_fuzzy = GCNConv(in_channels_fuzzy, hidden_channels)
        self.conv2_fuzzy = GCNConv(hidden_channels, out_channels)

        # Fully connected layers to combine embeddings and demographic data
        self.fc1 = nn.Linear(3 * out_channels + demographic_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # For binary classification

    def forward(self, data_desikan, data_destrieux, data_fuzzy):
        # Desikan branch
        x1 = torch.relu(self.conv1_desikan(data_desikan.x, data_desikan.edge_index))
        x1 = global_mean_pool(torch.relu(self.conv2_desikan(x1, data_desikan.edge_index)), data_desikan.batch)

        # Destrieux branch
        x2 = torch.relu(self.conv1_destrieux(data_destrieux.x, data_destrieux.edge_index))
        x2 = global_mean_pool(torch.relu(self.conv2_destrieux(x2, data_destrieux.edge_index)), data_destrieux.batch)

        # Fuzzy branch
        x3 = torch.relu(self.conv1_fuzzy(data_fuzzy.x, data_fuzzy.edge_index))
        x3 = global_mean_pool(torch.relu(self.conv2_fuzzy(x3, data_fuzzy.edge_index)), data_fuzzy.batch)

        # Combine all the embeddings
        combined = torch.cat([x1, x2, x3, data_desikan.demographic], dim=1)

        # Fully connected layers
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
