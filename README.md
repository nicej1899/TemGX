# TemGX: Generating Temporal Graph Explanations for Temporal Graph Neural Networks
## Overview

This repository provides the official implementation of TemGX, a method for generating temporal graph explanations of predictions made by temporal graph neural networks (TGNNs). TemGX features a bi-level framework that extracts instance-level explanations (local subgraphs) and then learns a global dynamic Bayesian network to capture the temporal evolution of these explanatory nodes.

## Requirements
Python 3.7+
PyTorch, torch_geometric
numpy, pandas, scikit-learn
pgmpy
json, time, random, os (standard libraries)
Install via:
```
pip install torch torch_geometric numpy pandas scikit-learn pgmpy
```
Usage
Prepare  Dataset

Ensure you have a function get_metr_la_dataset() (or similar) returning the temporal features and edge indices.
Train or load a pre-trained DCRNN model checkpoint.
Run Inner Layer
Execute ```temgx_inner.py``` to build the DBN
Update the checkpoint path and dataset loading as needed.
Execute the script  to generate json file.
Run Outer Layer

Point the script to the generated JSON file .
Execute ```temgx_upper.py``` to build the DBN, learn structures/parameters, and optionally retrieve CPDs.
Notes
You can modify hyperparameters (L, epsilon, window_size, discretization bins, etc.) to suit your dataset.
The DBN edges and Shapley-based explanations can be inspected or visualized for further analysis.
