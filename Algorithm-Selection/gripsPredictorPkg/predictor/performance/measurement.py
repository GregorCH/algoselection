import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from .. import logger
from .. import config
from .metric import sgm
from .visualisation import plotter

# set up logging properties
log = logger._Logger.get_logger(__name__) # set module name for logging

def measure_performance_multiple_files(
                        actual_data_type = 'time',
                        npredictions = None,
                        calculate_incrementally = True,
                        file_regexs = ['*.csv']):

    '''Calculate performance values based on shifted geometric mean for a group
    of CSV files produced by a same classifier runned multiple times with different
    seeds.

    For each file in a group of CSV files, function :py:meth:`~predictor.performance.measurement.measure_performance`
    is called and results are averaged. As a result, this function for each file
    group creates a file with following information which can be used for plotting:

        - x - an array of values indicating number of algorithms in predicted portfolio
        - y - an array of averaged values explained in previous paragraf for each
              value of x
        - yerr - an array of values representing standard error of corresponding y
                 calculated over all executions of same classifier with different
                 seeds
        - ystd - an array of standard deviations of y

    File is written on the location pointed by configuration parameter :py:attr:`~predictor.config.Config.plot_data_dir`
    with a same name as the name of the classifier whose output was used to produce
    results. Along with result file, additional file with list of predictors for
    which results are produced is updated with entry for new result file, or created
    if it does not exist. Additional file is by default named ``plots_list.yml``. Name
    can be changed through :py:attr:`~predictor.config.Config.plots_list_filename` configuration parameter.
    By default, function automatically indentifies all different file groups in
    directory and produces resulting files for each of them.

    :param actual_data_type: Determine if time or primal-dual integral will be
        used for performance calculation. Parameter can have values ``time`` or ``pdi``.
    :param npredictions: Determine how many algorithms from predicted portfolio
        will be included in performance calculation.
        Example: for npredictions = 2 first two algorithms from portfolio will be
        used when calculating performance for each problem instance.
    :param calculate_incrementally: If set to True, function calculates an array
        of performance values for first N algorihms where N is [1, npredictions]
        interval.
    :param file_regexs: Determine which files will be selected for calculation.
        By default all CSV files in directory pointed by :py:attr:`~predictor.config.Config.predicted_data_dir`
        are selected. If you want to proces just certain group of files you can
        do that by passing list of regular expressions for **globl** syntax.
        For example, if you want to select just group of files starting with
        "KNeighbours" and "RandomForest" you would create following list:
        ``["KNeighbours*", "RandomForest*"]``. This will result with two result files
        with data for KNeighbours and RandomForest classifiers separately.
    '''

    # TODO check if required configuration parameters are set

    PREDICTED_DATA_DIRPATH = config.global_config.predicted_data_dir

    predicted_data_files = []
    [ \
        predicted_data_files.extend(glob.glob( os.path.join(PREDICTED_DATA_DIRPATH, regex)) ) \
        for regex in file_regexs \
    ]
    nfiles = len(predicted_data_files)    # count total number of files for logger
    n = 0

    # find all unique filenames that will be used for grouping
    # each filename represents one prediction model used
    unique_file_groups = set()

    # discover all different classifiers among the files selected by regex list
    for file in predicted_data_files:
        filename = os.path.basename(file)   # get just filename from absolute path
        unique_file_groups.add( filename[ : filename.rfind('_') ] )

    labels = []
    # iterate over all result files with different seeds and calculating
    # aritmetic mean over them for each discovered classifier
    for file_group in unique_file_groups:
        log.info('Processing files for {:s} file group'.format(file_group))
        y_data_filegroup = []
        files_in_group = glob.glob(
            os.path.join(PREDICTED_DATA_DIRPATH, file_group) + '*.csv',
        )

        #iterate over all files from the same group with different seeds
        for filename in files_in_group:
            n += 1
            log.info("[{:d}/{:d}] Loading file {:s}".format(n, nfiles, filename))
            # filename will be fullpath to the file, extracting just filename with extension
            filename = os.path.basename(filename)

            # update configuration to take into account time or pdi as a
            # actual measure for current classifier
            config.global_config.reload(actual_data_type)

            # acumulate results over different seeds
            y_data_filegroup.append(
                measure_performance(
                    actual_data_type = actual_data_type,
                    npredictions = npredictions,
                    predicted_data_filename = filename,
                    calculate_incrementally = True
                )
            )

        y = calculate_y(y_data_filegroup)
        x = calculate_x(y)
        yerr = calculate_yerr(y_data_filegroup)
        ystd = calculate_ystd(y_data_filegroup)

        plotter.Plotter.save_plot_data(file_group, x, y, yerr, ystd)

def calculate_y(ploting_data):
    '''Calculate y values vector from matrix where each row represents values
    returned by :py:meth:`~predictor.performance.measurement` function for each
    run of same classifier with different seed, by averaging values along x axis.

    :param ploting_data: matrix containing performance values for as rows for all
        each run of the same classifier.

    :return: The list of y values.
    '''
    ploting_data =  np.asarray(ploting_data)
    y = np.sum(ploting_data, axis = 0)
    return np.divide(y, len(ploting_data)).tolist()

def calculate_x(nalgorithms):
    '''Determine number of algorithms in portfolio and returns list representing
    range(1, number of algorithms).

    :param nalgorithms: an array of algorithms in portfolio.

    :return: The list of x values.
    '''
    return list( range(1, len(nalgorithms) + 1) )

def calculate_yerr(ploting_data):
    '''Calculates the standard error of the mean for multiple run of the same
    classifier with different seeds.

    :param ploting_data: matrix containing performance values for as rows for all
        each run of the same classifier.

    :return: The list of standard errors for corresponding y values.
    '''
    yerr = stats.sem( ploting_data, axis = 0 )
    return [ np.asscalar(elem) for elem in yerr ]

def calculate_ystd(ploting_data):
    '''Calcluates standard deviation for multiple run of the same classifier with
    different seeds.

    :param ploting_data: matrix containing performance values for as rows for all
        each run of the same classifier.

    :return: The list of standard deviation for corresponding y values.
    '''
    ystd = np.std( ploting_data, axis = 0 )
    return [ np.asscalar(elem) for elem in ystd ]

def measure_performance(predicted_data_filename,
                        actual_data_type = 'time',
                        npredictions = None,
                        measure_data_filename = None,
                        calculate_incrementally = False):
    '''
    Calculates performance metric for input CSV file containing list of predicted
    algorithms for each instance in a set.

    Resulting CSV file should be produced by evaluated classifier. Result file
    must not have headers, it should contain only rows where first cell of the row
    is name of an instance for whose algorithms are predicted, and other cells in
    the row contain algorithms in certain ranking.
    For each result file, function produces result based on ratio of two values:

        - shifted geometric mean of values returned by :py:meth:`~predictor.config.Config._calculate_predicted_value`
          function of :py:class:`~predictor.config.Config` class, for each instance from input CSV file. For
          more details about returned value, take a look at :py:meth:`~predictor.config.Config._calculate_predicted_value`
          function. (SGM_1)
        - shifted geometric mean of values returned by :py:meth:`~predictor.config.Config._calculate_actual_value`
          function of :py:class:`~predictor.config.Config` class, for each instance from input CSV file. For
          more details about returned value, take a look at :py:meth:`~predictor.config.Config_calculate_actual_value`
          function. (SGM_2)

    Final result of this function depends on algorithm that you want to compare your
    predictions with, named **referent** algorithm. Referent algorithm is determinated
    by :py:class:`~predictor.config.Config` class property called
    :py:meth:`~predictor.config.Config._algorithm_type_referent`, which is user
    defined property. In case that value for referent algorithm is ``best`` or
    ``naive`` final result is calculated as:

    .. math::

        res = \\frac{SGM_1}{SGM_2}

    otherwise as:

    .. math::

        res = 1 - \\frac{SGM_1}{SGM_2}


    :param actual_data_type: Can have values ``time`` or ``pdi`` which determine
        if time or primal-dual integral data will be used for calculation of the
        shifted geometric mean ratio.
    :param npredictions: Determines how many first predicted algorithms will be
        used for calculation of shifted geometic mean ratio. If ``npredictions`` is
        bigger of total number of predicted algorithms, value is trimmed to total
        number of predicted algorithms.
    :param calculate_incrementally: If set to True, function calculates an array
        of performance values for first N algorihms where N takes values from
        [1, npredictions] interval. If set to true, resulting array is returned,
        otherwise is returned just last element of the array.
    :param predicted_data_filename: Filename of CSV file with prediction results.
        Each row in the input file has to be in form instance_name : predicted_algorithm
        names without any additional header rows or columns.
    :param measure_data_filename: File name of CSV file with actual values.
        If ``actual_data_type = 'time'`` this parameter must contain name of file
        with actual execution time for each instance and algorithm, otherwise it
        has to contain name of file with primal-dual integral data.
    :return: List of performance metric value based on input file with predictions.
        If parameter ``calculate_incrementally`` is set to false, list contains
        single value.
    '''
    # TODO set params
    # TODO check required params

    alpha                       = config.global_config._alpha
    predicted_data_fullpath     = os.path.join(
                                    config.global_config.predicted_data_dir,
                                    predicted_data_filename)
    actual_data                 = config.global_config._actual_data
    algorithm_type_predicted    = config.global_config.algorithm_type_predicted

    try:
        predicted_data  = pd.read_csv(predicted_data_fullpath, header = None)
    except FileNotFoundError:
        log.critical("File %s is not found!".format(predicted_data_fullpath))
        sys.exit(1)

    # if number of algorithms that will be used for sgm calculation is not specified, all
    # algorithms will be considered
    end_index = npredictions if npredictions is not None else len(predicted_data.columns) - 1
    if end_index > len(predicted_data.columns) - 1:
        end_index = len(predicted_data.columns) - 1

    results = []
    # calculate SGM for increasing number of algorithms
    # (just for first one, than for first two and so on until all algorithms are considered for calculation)
    for i in range(1, end_index + 1):
        predicted_values = []
        best_values = []

        # calculate SGM for each instance in dataframe
        for index, row in predicted_data.iterrows():
            # extracting instance name
            instance = row[0]

            # extracting first i predicted algorithm names for current instance
            algorithms = ( predicted_data.ix[index, 1:i] ).tolist()

            # measured time values for predicted algorithms for specific instance
            # try except blok will be removed if models are tested only on feasible solutions
            try:
                predicted_values.append(
                    config.global_config._calculate_predicted_value( actual_data, instance, algorithms )
                )
            except KeyError:
                log.error("Pair ( %s, %s ) is not found while seraching for predicted values!".format(instance, algorithm))
                pass

            # reads shortest algorithm execution time for given instance
            try:
                best_values.append(
                    config.global_config._calculate_actual_value(actual_data, instance)
                )
            except KeyError:
                log.error("Pair ( %s, %s ) is not found while seraching for actual values!".format(instance, algorithm))
                pass

            if (algorithm_type_predicted == 'best'):
                result = sgm.shifted_geometric_mean(predicted_values, alpha) / sgm.shifted_geometric_mean(best_values, alpha)
            else:
                result = 1 - sgm.shifted_geometric_mean(predicted_values, alpha) / sgm.shifted_geometric_mean(best_values, alpha)

        results.append( result )
    return results if calculate_incrementally else [ results[-1] ]


#if __name__ == '__main__':
    # measure_performance function test
    #print(measure_performance(
    #        actual_data_type = 'pdi',
    #        predicted_data_filename = 'rfr_b_mip-all_reducedPCA_mip-PD_s0.csv',
    #        calculate_incrementally = True
    #    )
    #)

    # measure_performance_multiple_files function test
    #print(measure_performance_multiple_files(actual_data_type = 'pdi'))
