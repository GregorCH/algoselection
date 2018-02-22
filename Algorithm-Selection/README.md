# Algorithm Selection

This repository contains source code of package developed by GRIPS2017 Optimisation group working on an algorithm selection problem, as well as some results we got. Results were obtained by using the package on MIPDEV and REGIONS instance datasets for a subset of SCIP configurations (algorithms).

Repository structure:
- In ```gripsPredictorPkg``` you can find source code of the package produced for the algorithm selection problem called AlgorithmPredictor (AP).
- In ```results``` directory are placed files produced by the AP package. The directory contains following subdirectories:
    - ```data``` - Contains bunch of files produced by model training stage. Each file has a list of instances and, for each instance, a ranking of SCIP configurations. Files are separated into different subdirectories depending on model, feature and performance data type.
    - ```plot_data``` - Files in this directory represent the output of model accuracy evaluation stage. Files are organized in a similar fashion as in data directory. This data is further used for creating plots.
    - ```plots``` - Here you can find various plot showing the performance of AP's models.
- ```ForCleaning``` - This directory contains a bunch of things, such as "quick-and-dirty" scripts. Needs cleanup.
