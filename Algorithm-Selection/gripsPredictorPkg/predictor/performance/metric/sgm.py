import numpy as np

from predictor.errors import SGMNegativeValueError

def shifted_geometric_mean(iterable, alpha):
    """
    Calculates shifted geometric mean.

    :param iterable:        ordered collection of values which will be used for SGM calculation
    :param alpha:           value added to each element included in calculation of geometric mean

    :return                 value of shifted geometric mean
    """

    a = np.add(iterable, alpha);
    for value in a:
        if value < 0:
            raise SGMNegativeValueError(expression = value)

    a = np.log(a)
    return np.exp(a.sum() / len(a)) - alpha
