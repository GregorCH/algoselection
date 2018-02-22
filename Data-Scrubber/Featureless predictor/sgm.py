import numpy as np

def shifted_geometric_mean(iterable, alpha):
    """
    Calculates shifted geometric mean.

    :param iterable:        ordered collection of values which will be used for SGM calculation
    :param alpha:           value added to each element included in calculation of geometric mean

    :return                 value of shifted geometric mean
    """

    a = np.add(iterable, alpha);
    a = np.log(a)
    return np.exp(a.sum() / len(a)) - alpha
