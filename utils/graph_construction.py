import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

def compute_anatomical_proximity(roi_coordinates):
    """Compute anatomical proximity between brain regions using Euclidean distance."""
    proximity_matrix = squareform(pdist(roi_coordinates, metric='euclidean'))
    proximity_matrix = (proximity_matrix - proximity_matrix.min()) / (proximity_matrix.max() - proximity_matrix.min())
    return proximity_matrix

def compute_correlation(roi_features):
    """Compute correlation between ROI features (e.g., time series or intensity)."""
    N = roi_features.shape[0]
    correlation_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            correlation_matrix[i, j] = pearsonr(roi_features[i, :], roi_features[j, :])[0]
            correlation_matrix[j, i] = correlation_matrix[i, j]
    correlation_matrix = (correlation_matrix - correlation_matrix.min()) / (correlation_matrix.max() - correlation_matrix.min())
    return correlation_matrix

def combine_proximity_and_correlation(proximity_matrix, correlation_matrix, alpha=0.5):
    """Combine anatomical proximity and correlation matrices to form the final weighted edges."""
    combined_matrix = alpha * proximity_matrix + (1 - alpha) * correlation_matrix
    return combined_matrix
