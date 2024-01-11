# GCN-for-Modeling-CO2RR-catalyst-High-Entropy-Alloys
In the data/input folder the 'HEA_properties.csv' file contains the element intrinsic properties. the rest of the csv files within this folder are brought/downloaded from the work by Pedersen et al. on high entropy alloys. [1]

The src/shared folder includes scripts for 1)constructing graphs from the csv files and their corresponding featurization and 2)defining the architecture of the GCN model.

The src/training folder includes scripts for training the GCN model on the constructed graphs (we trained four models with different initializations).

The src/explanation folder includes scripts for explaining the trained GCN models (and their predictions) in addition to ranking the importance of the node features.

The src/test_training_data folder tests the trained GCN models on the training data which helps to visualize the GCN predictions versus DFT values (figure 2 in our manuscript).

The src/test_testing_data folder tests the trained GCN models on the data not seen by them (it includes all the possible combinations of the elements) which helps to evaluate their robustness.

[1]: https://pubs.acs.org/doi/full/10.1021/acscatal.9b04343
