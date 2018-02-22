import os
import sys
import math
import unittest
from pandas.util.testing import assert_frame_equal

sys.path.append('/home/gorana/Projects/Algorithm-Selection/Performance_Measure/SGM/gripsPredictorPkg/')

from predictor.config import Config
from predictor.config import DefaultParameterValues as dpv

ASSETS_DIR = os.path.normpath(
                        os.path.join( os.path.dirname(__file__), 'assets' )
                    )

class TestConfig(unittest.TestCase):

    ACTUAL_TIME_EXAMPLE_FULLPATH = os.path.join(
        ASSETS_DIR, 'actual_time_example.csv'
    )
    ACTUAL_TIME_EXAMPLE_FULLPATH2 = os.path.join(
        ASSETS_DIR, 'actual_time_example2.csv'
    )
    ACTUAL_PDI_DATA_EXAMPLE_FULLPATH = os.path.join(
        ASSETS_DIR, 'actual_pdi_example.csv'
    )
    ACTUAL_TIME_BROKEN_EXAMPLE_FULLPATH = os.path.join(
        ASSETS_DIR, 'actual_time_example_nonexisting.csv'
    )

    def setUp(self):
        self.cfg = Config()


    def test_default_config(self):
        self.assertEqual( self.cfg.alpha_time, dpv.alpha_time )
        self.assertEqual( self.cfg.alpha_pdi, dpv.alpha_pdi )
        self.assertEqual( self.cfg.actual_time_data_fullpath, dpv.actual_time_data_fullpath )
        self.assertEqual( self.cfg.actual_pdi_data_fullpath, dpv.actual_pdi_data_fullpath )
        self.assertEqual( self.cfg.predicted_data_dir, dpv.predicted_data_dir )
        self.assertEqual( self.cfg.algorithm_type_predicted, dpv.algorithm_type_predicted )
        self.assertEqual( self.cfg.algorithm_type_referent, dpv.algorithm_type_referent )
        self.assertIsNotNone( self.cfg._calculate_actual_value )
        self.assertIsNotNone( self.cfg._calculate_predicted_value )
        self.assertIsNotNone( self.cfg._naive_time_best )
        self.assertIsNotNone( self.cfg._naive_pdi_best )
        self.assertEqual( self.cfg.plot_data_dir, dpv.plot_data_dir )
        self.assertEqual( self.cfg.plots_dir, dpv.plots_dir )
        self.assertEqual( self.cfg.plots_list_filename, dpv.plots_list_filename )
        self.assertEqual( self.cfg.plot_name, dpv.plot_name )
        self.assertEqual( self.cfg.plot_file_format, dpv.plot_file_format )
        self.assertEqual( self.cfg.finvestig_images_dir, dpv.finvestig_images_dir )
        self.assertEqual( self.cfg.finvestig_data_dir, dpv.finvestig_data_dir )
        self.assertEqual( self.cfg.finvestig_results_dir, dpv.finvestig_results_dir )
        self.assertEqual( self.cfg.models_images_dir, dpv.models_images_dir )
        self.assertEqual( self.cfg.models_data_dir, dpv.models_data_dir )
        self.assertEqual( self.cfg.models_results_dir, dpv.models_results_dir )
        self.assertEqual( self.cfg.models_test_dir, dpv.models_test_dir )


    def test__init_actual_data(self):
        alpha = 10
        actual_data_fullpath = self.ACTUAL_TIME_EXAMPLE_FULLPATH2

        [actual_data, min] = self.cfg._init_actual_data(actual_data_fullpath, alpha)
        self.assertTrue( math.isclose(min, 3, rel_tol = 1e-13) ) # because of possible number precision error


    def test__init_actual_data_invalid_fullpath(self):
        actual_data_fullpath = self.ACTUAL_TIME_BROKEN_EXAMPLE_FULLPATH
        with self.assertRaises(SystemExit) as e1:
            with self.assertRaises(FileNotFoundError):
                self.cfg._init_actual_data(actual_data_fullpath, 10)
            self.assertEqual(e1.exception.code, 1)


    def test_set_parameter_alpha_time(self):
        self.cfg.set_parameter('alpha_time', 10)
        self.assertEqual( self.cfg.alpha_time, 10 )


    def test_set_parameter_alpha_pdi(self):
        self.cfg.set_parameter('alpha_pdi', 1000)
        self.assertEqual( self.cfg.alpha_pdi, 1000 )


    def test_set_parameter_algorithm_type_predicted(self):
        self.cfg.set_parameter('algorithm_type_predicted', 'some random value')
        self.assertEqual( self.cfg.algorithm_type_predicted, 'some random value' )
        self.assertIsNotNone( self.cfg._calculate_predicted_value )


    def test_set_parameter_algorithm_type_referent(self):
        self.cfg.set_parameter('algorithm_type_referent', 'best')
        self.assertEqual( self.cfg.algorithm_type_referent, 'best' )
        self.assertIsNotNone( self.cfg._calculate_actual_value )


    def test_set_parameter_actual_time_data_fullpath(self):
        self.cfg.set_parameter('alpha_time', 10)
        self.cfg.set_parameter(
            'actual_time_data_fullpath',
            self.ACTUAL_TIME_EXAMPLE_FULLPATH
        )

        self.assertIs(self.cfg.alpha_time, 10)
        self.assertEqual(
            self.cfg.actual_time_data_fullpath,
            self.ACTUAL_TIME_EXAMPLE_FULLPATH
        )
        self.assertIsNotNone(self.cfg.actual_time_data)
        self.assertIsNotNone(self.cfg._naive_time_best)


    def test_set_parameter_actual_time_data_fullpath_missing_alpha(self):
        alpha_old = self.cfg.alpha_time
        self.cfg.alpha_time = None
        with self.assertRaises(ValueError):
            self.cfg.set_parameter(
                'actual_time_data_fullpath',
                self.ACTUAL_TIME_EXAMPLE_FULLPATH
            )
        self.cfg.alpha_time = alpha_old


    def test_set_parameter_actual_pdi_data_fullpath(self):
        self.cfg.set_parameter('alpha_pdi', 1000)
        self.cfg.set_parameter(
            'actual_pdi_data_fullpath',
            self.ACTUAL_PDI_DATA_EXAMPLE_FULLPATH
        )

        self.assertIs(self.cfg.alpha_pdi, 1000)
        self.assertEqual(
            self.cfg.actual_pdi_data_fullpath,
            self.ACTUAL_PDI_DATA_EXAMPLE_FULLPATH
        )
        self.assertIsNotNone(self.cfg.actual_pdi_data)
        self.assertIsNotNone(self.cfg._naive_pdi_best)


    def test_set_parameter_actual_pdi_data_fullpath_missing_alpha(self):
        alpha_old = self.cfg.alpha_pdi
        self.cfg.alpha_pdi = None
        with self.assertRaises(ValueError):
            self.cfg.set_parameter(
                'actual_pdi_data_fullpath',
                self.ACTUAL_PDI_DATA_EXAMPLE_FULLPATH
            )
        self.cfg.alpha_pdi = alpha_old


    def test_set_parameter_plots_list_filename(self):
        new_value = 'custom_plots_list_name'

        self.cfg.set_parameter('plots_list_filename', new_value)
        self.assertEqual(self.cfg.plots_list_filename, new_value)


    def test_set_parameter_plot_name(self):
        new_value = 'custom_plot_name'

        self.cfg.set_parameter('plot_name', new_value)
        self.assertEqual(self.cfg.plot_name, new_value)


    def test_set_parameter_show_plot(self):
        new_value = True

        self.cfg.set_parameter('show_plot', new_value)
        self.assertEqual(self.cfg.show_plot, new_value)


    def test_set_parameter_plot_file_format(self):
        new_value = 'png'

        self.cfg.set_parameter('plot_file_format', new_value)
        self.assertEqual(self.cfg.plot_file_format, new_value)


    def test_reload_with_time_data(self):
        self.assertIsNone(self.cfg._alpha)
        self.assertIsNone(self.cfg._actual_data)

        # set time related data
        self.cfg.set_parameter('alpha_time', 10)
        self.cfg.set_parameter('actual_time_data_fullpath', self.ACTUAL_TIME_EXAMPLE_FULLPATH)

        self.cfg.reload('time')
        self.assertIs(self.cfg._alpha, 10)
        self.assertIsNotNone(self.cfg._actual_data)
        assert_frame_equal(self.cfg._actual_data, self.cfg.actual_time_data)


    def test_reload_with_pdi_data(self):
        self.assertIsNone(self.cfg._alpha)
        self.assertIsNone(self.cfg._actual_data)

        # set pdi related data
        self.cfg.set_parameter('alpha_pdi', 1000)
        self.cfg.set_parameter('actual_pdi_data_fullpath', self.ACTUAL_PDI_DATA_EXAMPLE_FULLPATH)

        self.cfg.reload('pdi')
        self.assertIs(self.cfg._alpha, 1000)
        self.assertIsNotNone(self.cfg._actual_data)
        assert_frame_equal(self.cfg._actual_data, self.cfg.actual_pdi_data)


if __name__ == '__main__':
    unittest.main()
