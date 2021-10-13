import math

import numpy as np

from toolbox.estimator.estimator import Estimator


def lin_CE(Yb, Z):
    # Yb (output), Z (input) are of shape (num timepoints, num variables)
    Am = np.linalg.lstsq(Z, Yb, rcond=None)[0]
    Yp = Z @ Am
    Up = Yb - Yp
    S = np.cov(Up.T)
    if S.ndim == 0:
        S = np.var(Up.T)
        detS = S
    else:
        detS = np.linalg.det(S)
    N = Yb.shape[1]
    ce = 0.5 * np.log(detS) + 0.5 * N * np.log(2 * np.pi * np.exp(1))
    return ce


class LinearEstimator(Estimator):

    pi = math.pi
    e = math.e
    log2pie = np.log(2 * pi * e)

    def __init__(self):
        self.type = "linear"

    def estimate_cmi(self, y, x0, y0):
        y = y.T
        x0 = x0.T
        y0 = y0.T
        H_Y_Y0 = lin_CE(y, y0)
        X0Y0 = np.concatenate((x0, y0), axis=1)
        H_Y_X0Y0 = lin_CE(y, X0Y0)
        cmi = H_Y_Y0 - H_Y_X0Y0
        # print(cmi, H_Y_Y0, H_Y_X0Y0)
        return cmi

    def estimate_entropy(self, x):
        # X is of shape (num var, num timepoints)
        covX = np.cov(x)
        if covX.ndim == 0:
            covX = np.var(x)
            det_covX = covX
            N = 1
        else:
            det_covX = np.linalg.det(covX)
            N = x.shape[0]
        h = 0.5 * np.log(det_covX) + 0.5 * N * LinearEstimator.log2pie
        return h
