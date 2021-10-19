"""Local O information computation.
    Work in progress.
 """
import math
import numpy as np

import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm

from toolbox.states_probabilities import StatesProbabilities, ProbabilityEstimator


def local_o_of_state(state, state_probability: ProbabilityEstimator):
    n_variables = state.shape[0]
    state = state.to_list()
    system_probability = state_probability.get_probability(list(range(n_variables)), state)
    local_o = (n_variables - 2) * math.log2(system_probability)
    for index_variable in range(n_variables):
        variable_list = [index_variable]
        no_variable_list = list(range(n_variables))
        no_variable_list.remove(index_variable)
        data_var = [state[index_variable]]
        variable_probability = state_probability.get_probability(variable_list, data_var)
        data_no_var = state.copy()
        data_no_var.pop(index_variable)
        no_variable_probability = state_probability.get_probability(no_variable_list, data_no_var)
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

        n_data_points = data_table.shape[0]
        local_os = []
        for data_point in tqdm(range(n_data_points)):
            state = data_table.iloc[data_point]
            local_o = local_o_of_state(state, self.probability_estimator)
            local_os.append(local_o)
        return local_os
