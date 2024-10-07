import numpy as np
import torch
from scipy import stats

def perform_t_tests_on_embeddings(combined_embeddings, group_labels):
    """
    Performs t-tests on node embeddings to identify significant brain regions.
    """
    # Assume group 0: SI/SA MDD, group 1: SI/SA no MDD
    group_0_indices = np.where(group_labels == 0)[0]
    group_1_indices = np.where(group_labels == 1)[0]

    # Perform t-tests for each brain region (node)
    p_values = []
    for node in range(combined_embeddings.shape[1]):
        t_stat, p_value = stats.ttest_ind(combined_embeddings[group_0_indices, node],
                                          combined_embeddings[group_1_indices, node])
        p_values.append(p_value)

    # Apply correction for multiple comparisons (Bonferroni correction)
    significant_nodes = np.where(np.array(p_values) < 0.05 / combined_embeddings.shape[1])[0]
    
    print(f"Significant brain regions (nodes): {significant_nodes}")
    return significant_nodes, p_values

# Load saved embeddings and labels
combined_embeddings = torch.load('combined_embeddings.pt')
group_labels = np.load('group_labels.npy')  # Assuming group labels are preprocessed and saved

# Perform t-test analysis
significant_nodes, p_values = perform_t_tests_on_embeddings(combined_embeddings, group_labels)
