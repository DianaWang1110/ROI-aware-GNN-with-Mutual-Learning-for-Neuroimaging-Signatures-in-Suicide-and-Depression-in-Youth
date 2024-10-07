import torch
from torch import optim, nn
from data_loader import load_data  # Custom data loader
from models.mutual_learning_gcn import MutualLearningGCN
from utils.evaluation import perform_ttest

# Load datasets for Desikan, Destrieux, and Fuzzy parcellations
data_desikan, data_destrieux, data_fuzzy, labels = load_data()

# Initialize the GCN model
model = MutualLearningGCN(in_channels_desikan=data_desikan.num_features,
                          in_channels_destrieux=data_destrieux.num_features,
                          in_channels_fuzzy=data_fuzzy.num_features,
                          hidden_channels=32, out_channels=16, demographic_dim=3)

# Optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model
def train_mutual_learning_gcn(model, data_desikan, data_destrieux, data_fuzzy, optimizer, criterion, labels, epochs=100):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out, embedding_desikan, embedding_destrieux, embedding_fuzzy = model(data_desikan, data_destrieux, data_fuzzy)

        # Compute loss
        loss = criterion(out, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Run the training process
train_mutual_learning_gcn(model, data_desikan, data_destrieux, data_fuzzy, optimizer, criterion, labels)

# Save embeddings for further analysis
with torch.no_grad():
    _, embedding_desikan, embedding_destrieux, embedding_fuzzy = model(data_desikan, data_destrieux, data_fuzzy)

combined_embeddings = torch.cat([embedding_desikan, embedding_destrieux, embedding_fuzzy], dim=1).cpu().numpy()

# You can now pass these embeddings to the t-test analysis or save them
torch.save(combined_embeddings, 'combined_embeddings.pt')
