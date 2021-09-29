"""Local O information computation.
    Work in progress.
 """

import sys
import numpy as np
import itertools
from tqdm.auto import tqdm
from toolbox.gcmi import copnorm, ent_g
from toolbox.lin_est import lin_ent
from toolbox.utils import bootci, CombinationsManager, ncr


def exhaustive_local_o(ts, config):
    higher_order = config["higher_order"]
    estimator = config["estimator"]
    Xfull = copnorm(ts)
    n_variables, n_observations = Xfull.shape
    print("Timeseries details - Number of variables: ", str(n_variables), ", Number of timepoints: ",
          str(n_observations))
    print("Computing Oinfo using " + estimator + " estimator")
    X = Xfull
    maxsize = config["maxsize"]  # 5 # max number of variables in the multiplet
    n_best = config["n_best"]  # 10 # number of most informative multiplets retained
    nboot = config["nboot"]  # 100 # number of bootstrap samples
    alphaval = 0.05
    o_b = np.zeros((nboot, 1))
    print("NOT IMPLEMENTED YET")
    exit(1)
