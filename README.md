# TemGX: Generating Temporal Graph Explanations for Temporal Graph Neural Networks
## Overview

This repository provides the official implementation of TemGX, a method for generating temporal graph explanations of predictions made by temporal graph neural networks (TGNNs). TemGX features a bi-level framework that extracts instance-level explanations (local subgraphs) and then learns a global dynamic Bayesian network to capture the temporal evolution of these explanatory nodes.

## Requirements
```
	pip install torch torchvision torchaudio
	pip install torch-scatter
	pip install torch-sparse
	pip install torch-cluster
	pip install torch-spline-conv
	pip install torch-geometric
	pip install torch-geometric-temporal
	pip install numpy pandas scikit-learn
	pip install pgmpy
	pip install matplotlib
	pip install tensorboard

```


### Dataset
The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY) can be download from https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX and  https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph.
User should download the following files:

metr-la.h5
adj_mx.pkl and rename it to adj_mx_metr_la.pkl
distances_la_2012.csv and rename it to distances_metr_la.csv
graph_sensor_locations.csv and rename it to graph_sensor_locations_metr_la.csv
and copy them in the folder ./data/metr-la/.

Furthermore, the user should download the following files:

pems-bay-h5
adj_mx_bay.pkl and rename it to adj_mx_pems_bay.pkl
distances_bay_2017.csv and rename it to distances_pems_bay.csv
graph_sensor_locations_bay.csv and rename it to graph_sensor_locations_pems_bay.csv
and copy them in the folder ./data/pems-bay/.


###Train or load a pre-trained DCRNN model checkpoint.

Train the model:
```python -m src.train```

Ensure you have a function get_metr_la_dataset() (or similar) returning the temporal features and edge indices.


### Run Inner Layer
Execute ```python -m temgx_inner.py``` to build the DBN
Update the checkpoint path and dataset loading as needed.
Execute the script  to generate json file.

### Run Upper Layer

Point the script to the generated JSON file .
Execute ```python -m temgx_upper.py``` to build the DBN, learn structures/parameters, and optionally retrieve CPDs.
Notes
You can modify hyperparameters (L, epsilon, window_size, discretization bins, etc.) to suit your dataset.
The DBN edges and Shapley-based explanations can be inspected or visualized for further analysis.

### Full version
The full version of the paper is https://github.com/nicej1899/TemGx-full-version
