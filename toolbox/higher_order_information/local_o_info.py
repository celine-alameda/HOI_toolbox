"""Local O information computation.
    Work in progress.
 """
import concurrent.futures
import math
import numpy as np

import pandas as pd
import scipy
from sklearn.utils import resample
from tqdm import tqdm

from toolbox.states_probabilities import StatesProbabilities, ProbabilityEstimator


def local_o_of_index(n_variables, index, probabilities: dict):
    system_probability = probabilities[tuple(range(n_variables))][index]
    local_o = (n_variables - 2) * math.log2(system_probability)
    for index_variable in range(n_variables):
        variable_list = tuple([index_variable])
        no_variable_list = list(range(n_variables))
        no_variable_list.remove(index_variable)
        no_variable_list = tuple(no_variable_list)
        variable_probability = probabilities[variable_list][index]
        no_variable_probability = probabilities[no_variable_list][index]
        local_o += (math.log2(variable_probability) - math.log2(no_variable_probability))
    return local_o


class LocalOHOI:

    def __init__(self, probability_estimator: ProbabilityEstimator):
        self.bootstrap = False  # todo remove this argument and associated logic
        self.probability_estimator = probability_estimator

    # todo add some check to see if baseline table has the same columns as data_table
    def exhaustive_local_o(self, data_table: pd.DataFrame):
        """Computes local o information and significance for all data states in data_table.
            Returns
            -------
                local_os : a dictionary of all local information associated with bootstrapped significance
                (local_o ci including 0 or not)
        """
        n_variables = data_table.shape[1]
        n_datapoints = data_table.shape[0]
        combinations_to_compute = variable_combinations(n_variables)
        probability_dict = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(compute_probability, data_table, combination): combination for combination
                             in combinations_to_compute}
            for future in concurrent.futures.as_completed(futures):
                combination = futures[future]
                try:
                    probabilities = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (combination, exc))
                else:
                    probability_dict[tuple(combination)] = probabilities

        local_os = []
        print("Computing entropies from probabilities...")
        for i in tqdm(range(n_datapoints)):
            local_os.append(local_o_of_index(n_variables, i, probability_dict))
        return local_os


def compute_probability(data_table, combination):
    data_necessary = data_table.iloc[:, combination]
    # transpose because the KDE is created with dimensions x datapoints
    # the data_necessary table is row x columns
    return scipy.stats.gaussian_kde(data_necessary.T).pdf(data_necessary.T)


def variable_combinations(n_variables):
    combinations = [list(range(n_variables))]
    for index_variable in range(n_variables):
        variable_list = [index_variable]
        no_variable_list = list(range(n_variables))
        no_variable_list.remove(index_variable)
        combinations.append(variable_list)
        combinations.append(no_variable_list)
    return combinations
