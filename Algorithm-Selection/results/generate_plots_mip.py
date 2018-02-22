import os
import sys

# change this to path to directory that contains predictor package
sys.path.append('/home/gorana/Desktop/grips2017/Algorithm-Selection/Performance_Measure/SGM')

from predictor.performance import measurement as performance_metric
from predictor.performance.visualisation import plotter

ROOT_DIR = '/home/gorana/Desktop/grips2017/Algorithm-Selection/Performance_Measure/SGM/results/mip' # putanja do korenskog data foldera

# path to directory containing configuration files for calculating performance metric
CONFIG_FILES_PATH = '/home/gorana/Desktop/config-files/mip'

DEFAULT_DATA_DIRPATH = 'data/'
DEFAULT_PLOT_DATA_DIRPATH = 'plot_data/'
DEFAULT_PLOTS_DIRPATH = 'plots/'

DEFAULT_STATIC_FEATURES = 'static_features'
DEFAULT_DYNAMIC_FEATURES = 'dynamic_features'

# generate plots data for mipdev (both versions with and without variance)

################################################################################
############################# DYNAMIC FEATURES #################################
################################################################################
########################### A_SCALED DYNAMIC (a_scaled_type_2)
# run on combination of static and dynamic features on mipdev set

# ### GENERATING DATA FOR PLOTS
# # generate plot data for dynamic features using pdi data
# print('Generating data for A_SCALED DYNAMIC...')
# performance_metric.set_configuration( \
#     config_file_path = os.path.join(CONFIG_FILES_PATH, 'a_scaled_dynamic_pdi_pdi_config.yml') \
# )
# performance_metric.measure_performance_multiple_files( \
#     actual_data_type = 'pdi' \
# )
# # generate plot data for dynamic features using time data
# performance_metric.set_configuration( \
#     config_file_path = os.path.join(CONFIG_FILES_PATH, 'a_scaled_dynamic_pdi_time_config.yml') \
# )
# performance_metric.measure_performance_multiple_files( \
#     actual_data_type = 'time' \
# )
# print('Done!')
#
# ### GENERATING PLOTS
# print('Generating plots for A_SCALED DYNAMIC...')
# p = plotter.Plotter()
# PLOT_DATA_DIRPATH = os.path.join(ROOT_DIR, DEFAULT_PLOT_DATA_DIRPATH, DEFAULT_DYNAMIC_FEATURES)
# PLOTS_DIRPATH = os.path.join(ROOT_DIR, DEFAULT_PLOTS_DIRPATH, DEFAULT_DYNAMIC_FEATURES)
#
# # trained on pdi, performance calculated on pdi
# p.plot_performance(
#     plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'a_scaled/pdi_pdi'), \
#     plot_dirpath = os.path.join(PLOTS_DIRPATH, 'a_scaled'), \
#     plot_name = 'a_scaled_dynamic_pdi_pdi' \
# )
# p.plot_performance(
#     plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'a_scaled/pdi_pdi'), \
#     plot_dirpath = os.path.join(PLOTS_DIRPATH, 'a_scaled'), \
#     plot_name = 'a_scaled_dynamic_pdi_pdi', \
#     plot_variance = True
# )
# # trained on pdi, performance calculated on time
# p.plot_performance(
#     plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'a_scaled/pdi_time'), \
#     plot_dirpath = os.path.join(PLOTS_DIRPATH, 'a_scaled'), \
#     plot_name = 'a_scaled_dynamic_pdi_time' \
# )
# p.plot_performance(
#     plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'a_scaled/pdi_time'), \
#     plot_dirpath = os.path.join(PLOTS_DIRPATH, 'a_scaled'), \
#     plot_name = 'a_scaled_dynamic_pdi_time', \
#     plot_variance = True
# )
# print('Done!')
# input('Press Enter to proceed to the next calculation...')
#
#
#
# ########################### B_ALL_FEATURES DYNAMIC (run on all features from mipdev feature set)
# # run on combination of static and dynamic features on mipdev set
#
# ### GENERATING DATA FOR PLOTS
# # generate plot data for dynamic features using pdi data
# print('Generating data for B_ALL_FEATURES DYNAMIC...')
# performance_metric.set_configuration( \
#     config_file_path = os.path.join(CONFIG_FILES_PATH, 'b_all_features_dynamic_pdi_pdi_config.yml') \
# )
# performance_metric.measure_performance_multiple_files( \
#     actual_data_type = 'pdi' \
# )
# # generate plot data for dynamic features using time data
# performance_metric.set_configuration( \
#     config_file_path = os.path.join(CONFIG_FILES_PATH, 'b_all_features_dynamic_pdi_time_config.yml') \
# )
# performance_metric.measure_performance_multiple_files( \
#     actual_data_type = 'time' \
# )
# print('Done!')
#
# ### GENERATING PLOTS
# p = plotter.Plotter()
#
# print('Generating plots for B_ALL_FEATURES DYNAMIC...')
# # trained on pdi, performance calculated on pdi
# p.plot_performance(
#     plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'b_all_features/pdi_pdi'), \
#     plot_dirpath = os.path.join(PLOTS_DIRPATH, 'b_all_features'), \
#     plot_name = 'b_all_features_pdi_pdi' \
# )
# p.plot_performance(
#     plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'b_all_features/pdi_pdi'), \
#     plot_dirpath = os.path.join(PLOTS_DIRPATH, 'b_all_features'), \
#     plot_name = 'b_all_features_pdi_pdi', \
#     plot_variance = True
# )
# # trained on pdi, calculated on time
# p.plot_performance(
#     plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'b_all_features/pdi_time'), \
#     plot_dirpath = os.path.join(PLOTS_DIRPATH, 'b_all_features'), \
#     plot_name = 'b_all_features_pdi_time' \
# )
# p.plot_performance(
#     plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'b_all_features/pdi_time'), \
#     plot_dirpath = os.path.join(PLOTS_DIRPATH, 'b_all_features'), \
#     plot_name = 'b_all_features_pdi_time', \
#     plot_variance = True
# )
# print('Done!')
# input('Press Enter to proceed to the next calculation...')
#
#
#
#
# ################################################################################
# ############################# STATIC FEATURES ##################################
# ################################################################################
# ########################### A_SCALED (scaling type 2)
# #
PLOT_DATA_DIRPATH = os.path.join(ROOT_DIR, DEFAULT_PLOT_DATA_DIRPATH, DEFAULT_STATIC_FEATURES)
PLOTS_DIRPATH = os.path.join(ROOT_DIR, DEFAULT_PLOTS_DIRPATH, DEFAULT_STATIC_FEATURES)

### GENERATING DATA FOR PLOTS
print('Generating data for A_SCALED STATIC...')
performance_metric.set_configuration( \
    config_file_path = os.path.join(CONFIG_FILES_PATH, 'a_scaled_static_pdi_pdi_config.yml') \
)
performance_metric.measure_performance_multiple_files( \
    actual_data_type = 'pdi' \
)
# generate plot data for dynamic features using time data
performance_metric.set_configuration( \
    config_file_path = os.path.join(CONFIG_FILES_PATH, 'a_scaled_static_pdi_time_config.yml') \
)
performance_metric.measure_performance_multiple_files( \
    actual_data_type = 'time' \
)
print('Done!')

## GENERATING PLOTS
p = plotter.Plotter()

print('Generating plots for A_SCALED STATIC...')
# trained on pdi, calculated on pdi
p.plot_performance(
    plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'a_scaled/pdi_pdi'), \
    plot_dirpath = os.path.join(PLOTS_DIRPATH, 'a_scaled'), \
    plot_name = 'a_scaled_static_pdi_pdi' \
)
p.plot_performance(
    plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'a_scaled/pdi_pdi'), \
    plot_dirpath = os.path.join(PLOTS_DIRPATH, 'a_scaled'), \
    plot_name = 'a_scaled_static_pdi_pdi', \
    plot_variance = True
)

# trained on pdi, calculated on time
p.plot_performance(
    plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'a_scaled/pdi_time'), \
    plot_dirpath = os.path.join(PLOTS_DIRPATH, 'a_scaled'), \
    plot_name = 'a_scaled_static_pdi_time' \
)
p.plot_performance(
    plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'a_scaled/pdi_time'), \
    plot_dirpath = os.path.join(PLOTS_DIRPATH, 'a_scaled'), \
    plot_name = 'a_scaled_static_pdi_time', \
    plot_variance = True
)
print('Done!')
input('Press Enter to proceed to the next calculation...')
#
#
#
# ########################### B_ALL_FEATURES STATIC
# # run on all static features on mipdev set
# ### GENERATING DATA FOR PLOTS
# print('Generating data for B_ALL_FEATURES STATIC...')
# performance_metric.set_configuration( \
#     config_file_path = os.path.join(CONFIG_FILES_PATH, 'b_all_features_static_pdi_pdi_config.yml') \
# )
# performance_metric.measure_performance_multiple_files( \
#     actual_data_type = 'pdi' \
# )
# # generate plot data for dynamic features using time data
# performance_metric.set_configuration( \
#     config_file_path = os.path.join(CONFIG_FILES_PATH, 'b_all_features_static_pdi_time_config.yml') \
# )
# performance_metric.measure_performance_multiple_files( \
#     actual_data_type = 'time' \
# )
# print('Done!')

### GENERATING PLOTS
p = plotter.Plotter()

print('Generating plots for B_ALL_FEATURES STATIC...')
p.plot_performance(
    plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'b_all_features/pdi_pdi'), \
    plot_dirpath = os.path.join(PLOTS_DIRPATH, 'b_all_features'), \
    plot_name = 'b_all_features_static_pdi_pdi' \
)
p.plot_performance(
    plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'b_all_features/pdi_pdi'), \
    plot_dirpath = os.path.join(PLOTS_DIRPATH, 'b_all_features'), \
    plot_name = 'b_all_features_static_pdi_pdi', \
    plot_variance = True
)
p.plot_performance(
    plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'b_all_features/pdi_time'), \
    plot_dirpath = os.path.join(PLOTS_DIRPATH, 'b_all_features'), \
    plot_name = 'b_all_features_static_pdi_time' \
)
p.plot_performance(
    plot_data_dirpath = os.path.join(PLOT_DATA_DIRPATH, 'b_all_features/pdi_time'), \
    plot_dirpath = os.path.join(PLOTS_DIRPATH, 'b_all_features'), \
    plot_name = 'b_all_features_static_pdi_time', \
    plot_variance = True
)
print('Done!')
