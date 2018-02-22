@echo off
python3 featureless-predictor.py regions_avg_time.csv time
python3 featureless-predictor.py regions_avg_PD.csv pdi
python3 featureless-predictor.py mipdev_feasible_avg_time.csv time
python3 featureless-predictor.py mipdev_avg_PD.csv pdi