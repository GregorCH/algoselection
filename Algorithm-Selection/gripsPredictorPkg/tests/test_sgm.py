import sys, unittest, math
from scipy.stats.mstats import gmean

from predictor.performance.metric import sgm
from predictor.errors import SGMNegativeValueError

sys.path.append('/home/gorana/Projects/Algorithm-Selection/Performance_Measure/SGM/gripsPredictorPkg/')

class TestSGM(unittest.TestCase):

    def test_all_positive(self):
        values = [1.0, 2.0, 3.0, 4.0]
        alpha = 10

        res = sgm.shifted_geometric_mean(values, alpha)

        self.assertTrue( math.isclose(res, 2.449770044, rel_tol = 1e-9) )

    def test_all_positive_with_alpha(self):
        values = [-1.0, -2.0, -3.0, -4.0]
        alpha = 10

        res = sgm.shifted_geometric_mean(values, alpha)

        self.assertTrue( math.isclose(res, -2.584414498, rel_tol = 1e-9) )

    def test_negative_values(self):
        values = [-11.0, -22.0, -33.0, -44.0]
        alpha = 10

        with self.assertRaises(SGMNegativeValueError) as e:
            sgm.shifted_geometric_mean(values, alpha)
            self.assertEqual(e.exception.expression, -1.0)
