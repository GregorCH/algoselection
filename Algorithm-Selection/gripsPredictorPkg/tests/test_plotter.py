import os
import sys
import unittest

sys.path.insert(0, '/home/gorana/Projects/Algorithm-Selection/Performance_Measure/SGM/gripsPredictorPkg/')

from predictor.config import Config, DefaultParameterValues as dpv
from predictor.performance.visualisation import plotter

ASSETS_DIR = os.path.normpath(
                        os.path.join( os.path.dirname(__file__), 'assets' )
                    )

class TestPlotter(unittest.TestCase):

    def setUp(self):
        self.cfg = Config()
        self.pltr = plotter.Plotter()

    # one curve is the best for all portfolios
    def test_choose_best_curves_portfolio_size_12(self):
        self.pltr.plot_collection.clear()
        curve1 = {
            'plot_name': 'Classifier1',
            'x_data': [1, 2, 3],
            'y_data': [1.1, 1.1, 1.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve1)
        curve2 = {
            'plot_name': 'Classifier2',
            'x_data': [1, 2, 3],
            'y_data': [0.1, 0.1, 0.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve2)
        curve3 = {
            'plot_name': 'Classifier3',
            'x_data': [1, 2, 3],
            'y_data': [3.1, 3.1, 3.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve3)

        result = self.pltr.choose_best_curves(portfolio_sizes = range(1, 3))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], curve2)

    def test_choose_best_curves_portfolio_size_all(self):
        self.pltr.plot_collection.clear()
        # best for three algorithms in portfolio
        curve1 = {
            'plot_name': 'Classifier1',
            'x_data': [1, 2, 3],
            'y_data': [5.1, 5.1, 1.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve1)
        # best for one algorithm in portfolio
        curve2 = {
            'plot_name': 'Classifier2',
            'x_data': [1, 2, 3],
            'y_data': [0.1, 5.1, 5.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve2)
        # best for two algorithms in portfolio
        curve3 = {
            'plot_name': 'Classifier3',
            'x_data': [1, 2, 3],
            'y_data': [5.1, 3.1, 5.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve3)

        result = self.pltr.choose_best_curves(portfolio_sizes = range(1,4))

        self.assertEqual(len(result), 3)
        self.assertEqual(result, [curve2, curve3, curve1])

    def test_choose_best_curves_portfolio_size_1(self):
        self.pltr.plot_collection.clear()
        # best for three algorithms in portfolio
        curve1 = {
            'plot_name': 'Classifier1',
            'x_data': [1, 2, 3],
            'y_data': [5.1, 5.1, 1.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve1)
        # best for one algorithm in portfolio
        curve2 = {
            'plot_name': 'Classifier2',
            'x_data': [1, 2, 3],
            'y_data': [0.1, 5.1, 5.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve2)
        # best for two algorithms in portfolio
        curve3 = {
            'plot_name': 'Classifier3',
            'x_data': [1, 2, 3],
            'y_data': [5.1, 3.1, 5.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve3)

        result = self.pltr.choose_best_curves(portfolio_sizes = [1])

        self.assertEqual(len(result), 1)
        self.assertEqual(result, [curve2])

    def test_choose_best_curves_portfolio_size_2(self):
        self.pltr.plot_collection.clear()
        # best for three algorithms in portfolio
        curve1 = {
            'plot_name': 'Classifier1',
            'x_data': [1, 2, 3],
            'y_data': [5.1, 5.1, 1.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve1)
        # best for one algorithm in portfolio
        curve2 = {
            'plot_name': 'Classifier2',
            'x_data': [1, 2, 3],
            'y_data': [0.1, 5.1, 5.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve2)
        # best for two algorithms in portfolio
        curve3 = {
            'plot_name': 'Classifier3',
            'x_data': [1, 2, 3],
            'y_data': [5.1, 3.1, 5.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve3)

        result = self.pltr.choose_best_curves(portfolio_sizes = [2])

        self.assertEqual(len(result), 1)
        self.assertEqual(result, [curve3])

    def test_choose_best_curves_portfolio_size_3(self):
        self.pltr.plot_collection.clear()
        # best for three algorithms in portfolio
        curve1 = {
            'plot_name': 'Classifier1',
            'x_data': [1, 2, 3],
            'y_data': [5.1, 5.1, 1.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve1)
        # best for one algorithm in portfolio
        curve2 = {
            'plot_name': 'Classifier2',
            'x_data': [1, 2, 3],
            'y_data': [0.1, 5.1, 5.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve2)
        # best for two algorithms in portfolio
        curve3 = {
            'plot_name': 'Classifier3',
            'x_data': [1, 2, 3],
            'y_data': [5.1, 3.1, 5.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve3)

        result = self.pltr.choose_best_curves(portfolio_sizes = [3])

        self.assertEqual(len(result), 1)
        self.assertEqual(result, [curve1])

    def test_choose_best_curve_portfolio_size_undefined(self):
        self.pltr.plot_collection.clear()
        curve1 = {
            'plot_name': 'Classifier1',
            'x_data': [1, 2, 3],
            'y_data': [5.1, 5.1, 1.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve1)
        curve2 = {
            'plot_name': 'Classifier2',
            'x_data': [1, 2, 3],
            'y_data': [0.1, 5.1, 5.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve2)
        curve3 = {
            'plot_name': 'Classifier3',
            'x_data': [1, 2, 3],
            'y_data': [5.1, 3.1, 5.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve3)
        curve4 = {
            'plot_name': 'Classifier3',
            'x_data': [1, 2, 3],
            'y_data': [10.1, 10.1, 10.1],
            'yerr_data': [],
            'ystd_data': []
        }
        self.pltr.plot_collection.append(curve4)

        result = self.pltr.choose_best_curves(portfolio_sizes = [])

        self.assertEqual(len(result), 4)
        self.assertEqual(result, [curve1, curve2, curve3, curve4])

if __name__ == '__main__':
    unittest.main()
