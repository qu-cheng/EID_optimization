# EID_optimization
This repository contains data and code for reproducing the analyses in **Network topology outweighs emergence probability in surveillance sentinel selection**
## Data
This folder contains **four empirical network datasets**, each stored in `.gml` format.  
These networks are used as real-world test cases for evaluating surveillance strategies.
- Each `.gml` file represents an empirical contact or interaction network.
- All analyses in the paper that refer to empirical networks are based on these files.
- Networks can be loaded directly using `networkx.read_gml()`.
## Visualization
This folder contains **all scripts used for generating figures** in the manuscript, including:
- Performance comparison plots
- Feature importance visualizations
- Sensitivity analysis figures
- incomplete-data performance plots
## Code
- `01_network_generator.py`: Generates synthetic networks with controlled topological properties.
- `02_generating_training_data.py`: Simulates epidemic outbreaks on networks and constructs training datasets for surveillance site selection models.
- `03_genetic_algorithm.py`: Implements a genetic algorithm to identify near-optimal sets of surveillance sites by maximizing early detection performance.
- `04_RFSM_and_importance.py`: Trains a random forest–based surrogate model and quantifies the importance of network and node features in site selection.
- `05_sensitivity_analyses.py`: Performs sensitivity analyses to evaluate the relative contributions of node characteristics.
- `06_performance_with_incomplete_data.py`: Assesses surveillance performance under incomplete network structure observation.
