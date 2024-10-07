# Neuroimaging Signatures of Structural MRI in SI/SA MDD vs Non-MDD using ROI-Aware GCN

This project utilizes Graph Neural Networks (GNNs), particularly ROI-aware Graph Convolutional Networks (GCN), to identify neuroimaging signatures associated with suicide attempts and major depressive disorder (MDD). The analysis combines multiple structural MRI features such as cortical thickness, surface area, volume, and sulcal depth using various brain parcellation schemes (Desikan, Destrieux, Fuzzy Clustering).

## Features

- **Mutual Learning GCN**: The model learns from multiple parcellation schemes (Desikan, Destrieux, Fuzzy Clustering) to identify neurobiological markers.
- **Feature Extraction**: Structural MRI features (T1 intensity, cortical thickness, surface area, sulcal depth, volume) are merged across different parcellations.
- **Statistical Analysis**: Group comparisons between SI/SA with MDD vs without MDD are performed using node embeddings extracted from the GCN models, with significant regions identified through t-tests.

## Directory Structure

- **README.md**
- **requirements.txt**
- **data/**
  - `demo_data.csv`: Placeholder for the demographic data (ABCD data)
  - `roi_data.csv`: Placeholder for the structural MRI ROI data (ABCD data)
  - `edge_index_data.csv`: Placeholder for the graph edge index data
- **models/**
  - `__init__.py`: Initialize model package
  - `roi_gcn.py`: ROI-aware GCN model
  - `mutual_learning_gcn.py`: Mutual learning GCN for parcellations
- **utils/**
  - `__init__.py`: Initialize utils package
  - `data_processing.py`: Data cleaning and preparation utilities
  - `graph_construction.py`: Functions for graph construction
  - `evaluation.py`: Evaluation metrics and statistical tests (e.g., t-tests)
- **training.py**: Main training script
- **analysis.py**: Analysis and statistical tests (e.g., t-test)
- **data_loader.py**: Data loader for structural MRI and demographic data

## Dataset

We use structural neuroimaging data from the Adolescent Brain Cognitive Development (ABCD) study, focusing on T1-weighted MRI data for two groups: - SI/SA with MDD: Individuals with suicidal ideation or attempts who also meet criteria for major depressive disorder. - SI/SA without MDD: Individuals with suicidal ideation or attempts but without major depressive disorder. The structural data includes multiple features such as T1 intensity, cortical thickness, sulcal depth, surface area, and volume across different parcellations (Desikan, Destrieux, and fuzzy clustering).

## Models

* **ROI-aware GCN**: A graph neural network model designed to utilize region-specific information to learn embeddings for brain regions based on MRI features and demographic data.

- **Mutual Learning GCN**: A model that simultaneously trains on multiple parcellations (Desikan, Destrieux, fuzzy clustering) to enable better generalization and learning across scales.

## Data

- `demo_data.csv`: Contains demographic information (gender, age, marital status, race, education, income).
- `roi_data.csv`: Contains structural MRI features (T1 intensity, cortical thickness, surface area, sulcal depth, volume) for various brain regions.
- `edge_index_data.csv`: Contains adjacency matrices for the brain regions based on anatomical proximity and feature correlation.

## Usage

### Training the Model

The training includes three GCN models for each brain parcellation (Desikan, Destrieux, Fuzzy Clustering) and the node embeddings are combined for downstream analysis.

```bash
python training.py
```

### Analyzing Results

After training, use the `analysis.py` script to perform group comparisons and t-tests to identify significant brain regions.

```bash
python analysis.py
```

## Example Output

- **Embeddings**: Node embeddings for each brain region, generated from the GCN models.
- **Statistical Analysis**: Significant brain regions identified using t-tests, corrected for multiple comparisons.
