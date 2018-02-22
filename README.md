# algoselection

This repository contains code and data that has been used for training and analysis of algorithm selection methods for SCIP on benchmark data for publicly available Mixed Integer Programming (MIP) instances.

## Overview: Algorithm Selection

Algorithm Selection is the task to select a well-performing algorithm out of an available portfolio based on a feature description of an input MIP.
For this project, different emphasis settings of SCIP form the available portfolio. Feature data has been collected for two different sets of instances, the homogenous regions test set and the heterogenous combination of the different MIPLIB and Cor@l benchmarks.



## Structure of the repository

The repository is structured as follows.

1. In **Data-Scrubber** reside scripts to transform and filter the original, raw feature and performance data. The directory also contains code for a feature-independent algorithm selector that serves as a comparison for the more advanced methods. 
2. The directory **Feature-Investigator** contains scripts and plots to analyze the feature landscape.

## Authors

- David Haley: Data-Scrubber, Feature-Investigator
- Alexander Georges: Feature-Investigator




