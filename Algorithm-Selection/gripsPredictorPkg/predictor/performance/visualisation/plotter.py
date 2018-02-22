import os
import yaml as serializer
import matplotlib.pyplot as plt

from ... import logger
from ... import config

log = logger._Logger.get_logger(__name__) # set module name for logging
cfg = config.global_config

class Plotter:
    '''Plotter class contains methods for creating different plots based on the
    plots data files produced by ``measurement`` module. Class contains methods
    for plotting performance plots for algorithm portfolios, with or without
    variation. It also offers method for plotting error bars.
    '''

    def __init__(self):

        ## Collection of curves that will be plotted on the same plot. Every curve is
        #  represented by dictionary storing array of x axis values, array of y axis
        #  values and array of yerr values (used for plotting error bars).
        self.plot_collection = []

        ## Fullpath of the file that contains list of plot data files from the plot_data_dirpath
        #  direcory that will be used for plotting.
        self.plots_list_fullpath = os.path.join(cfg.plot_data_dir, cfg.plots_list_filename) + '.yml'

    @classmethod
    def update_plots_list(cls, plot_name):
        '''Appends new prediction model in plots list. If plots list exists it is
        updated by new `classifier_name: yes` pair, otherwise plots file is first
        created and then modified. Configuration parameter ``plot_data_dir`` determine
        location of plots list file, and ``plots_list_filename`` name of that file.

        :param plot_name: Name of the model that will be added to the list.
        '''
        plots_list_fullpath = os.path.join(cfg.plot_data_dir, cfg.plots_list_filename)
        with open(plots_list_fullpath, 'a') as output:
            output.write(plot_name + ": yes\n")

    def load_plots_list(self):
        '''Loads plots list from location pointed by ``plots_list_fullpath``
        class property.

        If plots list contain commented entries, or those where is explicitly stated
        that line should not be included on the plot (next to the line name is
        written string "no"), those lines are omitted.

        :return: List of plot names loaded from plots list that are going to be
            plotted.
        '''
        with open(self.plots_list_fullpath, 'r') as ymlfile:
            plot_list = serializer.load(ymlfile)

        plot_names = []
        for plot_name, status in plot_list.items():
            if status:
                plot_names.append( plot_name )
                self.plot_collection.append( self.load_plot_data(plot_name) )
        return plot_names

    @classmethod
    def save_plot_data(cls, plot_name, x_data, y_data, yerr_data, ystd_data,
                        plot_data_dir = None, plots_list_filename = None):
        '''Saves data for one classifier that later can be used for plotting.

        :param plot_name: Plot name. Usually name of classifier whose results were
            used for generating plot data.
        :param x_data: Array of numbers representing range between 1 and number
            of algorithms in portfolio.
        :param y_data: Array of values for performance plot.
        :param yerr_data: Array of standard measurement error values for error bar plot.
        :param ystd_data: Array of standard deviation values for displaying y data
            variations.
        :param plot_data_dirpath: Path to directory where plot data will be saved.
            If omitted, value of configuration parameter ``plot_data_dir`` is used.
        :param plots_list_filename: Name of the plots list file that will be updated.
            If omitted, value of configuration parameter ``plots_list_filename``
            is used.
        '''
        if plot_data_dir is not None:
            cfg.set_parameter('plot_data_dir', plot_data_dir)
        if plots_list_filename is not None:
            cfg.set_parameter('plots_list_filename', plots_list_filename)

        plot_data = {
            'plot_name' : plot_name,
            'x_data'    : x_data,
            'y_data'    : y_data,
            'yerr_data' : yerr_data,
            'ystd_data' : ystd_data
        }

        plot_data_fullpath = os.path.join(cfg.plot_data_dir, plot_name + ".pdat")
        with open(plot_data_fullpath, 'w') as output:
            serializer.dump(plot_data, output)
            cls.update_plots_list(plot_name)
        log.info("Performance metric for model " + plot_name + " saved on path " + plot_data_fullpath + "!")

    def load_plot_data(self, plot_name):
        '''Loads plot data from plot data file.

        :param plot_name: Name of the file that contains data to be read. It has
            to be passed without a file extension.
        :return: Loaded data in form of dictionary with x, y, yerr, ystd and
            plot_name as keys.
        '''
        plot_data_fullpath = os.path.join(cfg.plot_data_dir, plot_name + ".pdat")
        with open(plot_data_fullpath, 'r') as input:
            new_plot_data = serializer.load(input)
        return new_plot_data

    def choose_best_curves(self, portfolio_sizes):
        '''Choose curves with minimal value of y for each size of portfolio.

        :param portfolio_sizes: List of portfolio_sizes for which best curves will
            be chosen. For example, if portfolio_sizes = [1, 2, 5], function will
            determine which classifier performed the best for just one algorithm
            in portfolio and print just that line. Same thing will be done for
            portfolio_sizes of size 2 and 5 and up to three curves will be
            plotted (in case that all three curves are different). All other
            curves will be ommited.

        :return: Reduced list of curves for plotting.
        '''
        reduced_plot_collection = []

        if len(portfolio_sizes) == 0:
            return self.plot_collection

        for i in portfolio_sizes:
            # for each classifier get y value for portfolio size = i
            ydata = [ y_data['y_data'][i-1] for y_data in self.plot_collection ]

            # find best classifier by finding best (min) y value and then index
            # of that value. That index is same as an index of the best classifier
            # from classifier's collection.
            # FIXME: If multiple min values are found, just classifier that produced
            # first one will be included
            best_classifier = self.plot_collection[ydata.index(min(ydata))]
            reduced_plot_collection.append(best_classifier)

        # remove same curves that appear multiple times
        curves_without_duplicates = []
        curve_names = []

        for curve in reduced_plot_collection:
            if curve['plot_name'] not in curve_names:
                curve_names.append(curve['plot_name'])
                curves_without_duplicates.append(curve)

        return curves_without_duplicates

    def plot_performance(
        self,
        plot_data_dirpath = cfg.plot_data_dir,
        plot_dirpath = cfg.plots_dir,
        plots_list_filename = cfg.plots_list_filename + '.yml',
        plot_name = cfg.plot_name,
        plot_variance = False,
        plot_best_curves = [],
        show_plot = cfg.show_plot):

        '''Plots performance plot for a set of classifiers specified in given
        plots list file. Generated plot is automaticaly saved in directory pointed
        by function parameter ``plots_dir``.

        :param plot_data_dirpath: Path to the directory where plot data files are
            stored. If not specified it is set to default value.
        :param plot_dirpath: Path to the directory where created plot will be
            saved. If not specified it is set to default value.
        :param plots_list_filename: Name of the plots list file that is used to
            determine which plot data files will be loaded and plotted. It has
            to be passed without extension.
            If not specified method searches for file named named according to
            the parameter ``plots_list_filename`` from ``Config`` class.
        :param plot_name: Name of the created plot. If not specified, plot will
            be named after some of the loaded plot data files.
        :param plot_variance: If set to true performance plot will also show
            variance for y axis values.
        :param plot_best_curves: List of portfolio sizes for which curves with
            minimal value for y axis are chosen and plotted. All other curves
            that doesn't have minimum y value in any of portfolio sizes from the
            list, are not plotted even if they are marked to be plotted in the
            loaded plot list. If empty list is passed (which is default value)
            all curves marked for plotting from plot list are plotted.
        :param show_plot: Indicates if plot will be showed after creation. If not
            specified it is set to default value.
        '''

        #if not self.plot_collection:
        self.plot_collection.clear()
        self.load_plots_list()

        self.plot_collection = self.choose_best_curves(plot_best_curves)

        labels = []
        plt.figure()
        for plot_data in self.plot_collection:
            plot_name = plot_data['plot_name']
            x_data = plot_data['x_data']
            y_data = plot_data['y_data']
            ystd_data = plot_data['ystd_data']
            if plot_variance:
                plt.errorbar(x_data, y_data, ystd_data, marker = 'o', label = plot_name)
            else:
                plt.plot(x_data, y_data, marker = 'o', label = plot_name)
            labels.append(plot_name)

        plt.xlabel('Size of predicted portfolio')
        plt.ylabel('SGM ratio')
        plt.title('Portfolio performance')
        plt.legend(labels, prop = { 'size' : 8 })
        plt.grid(True)

        if plot_variance:
            file_name = cfg.plot_name + '_performance_with_variance.svg'
        else:
            file_name = cfg.plot_name + '_performance.svg'
        destination_path = os.path.join(cfg.plots_dir, file_name)
        plt.savefig(destination_path)
        log.info("Pefromance plot for " + plot_data['plot_name'] + " saved on " +
                     destination_path + "!")
        if show_plot:
            plt.show()

    def plot_error_bars(
        self,
        plot_data_dir = cfg.plot_data_dir,
        plots_dir = cfg.plots_dir,
        plots_list_filename = cfg.plots_list_filename,
        plot_name = cfg.plot_name,
        show_plot = cfg.show_plot):
        '''Plots error bar plots for set of classifiers specified in given plots
        list file.

        One error plot bar is created per one curve and automatically saved in
        directory pointed by function parameter ``plots_dir``.

        :param plot_data_dirpath: Path to the directory where plot data files are
            stored. If not specified it is set to default value.
        :param plot_dirpath: Path to the directory where created plot will be
            saved. If not specified it is set to default value.
        :param plots_list_filename: Name of the plots list file that is used to
            determine which plot data files will be loaded and plotted. It has
            to be passed without extension.
            If not specified method searches for file named named according to
            the parameter ``plots_list_filename`` from ``Config`` class.
        :param plot_name: Name of the created plot. If not specified, plot will
            be named after some of the loaded plot data files.
        :param show_plot: Indicates if plot will be showed after creation. If not
            specified it is set to default value.
        '''

        if not self.plot_collection:
            self.load_plots_list()

        for plot_data in self.plot_collection:
            plot_name = plot_data['plot_name']
            x_data = plot_data['x_data']
            y_data = plot_data['y_data']
            yerr_data = plot_data['yerr_data']

            plt.figure()
            plt.errorbar(x_data, y_data, yerr_data, marker = 'o')
            plt.title(plot_name + " error bar")
            destination_path = os.path.join(cfg.plots_dir, plot_name + '_error_bars.svg')
            plt.savefig(destination_path)
            log.info("Error bar plot for " + plot_data['plot_name'] + " saved on " +
                         destination_path + "!")
            if cfg.show_plot:
                plt.show()
