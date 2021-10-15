"""Local O information computation.
    Work in progress.
 """
import math
import numpy as np

import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm

from toolbox.states_probabilities import StatesProbabilities


def local_o_state(data, state_probability):
    n_variables = data.shape[0]
    data = data.to_list()
    system_probability = state_probability.get_probability(list(range(n_variables)), data)
    local_o = (n_variables - 2) * math.log2(system_probability)
    for index_variable in range(n_variables):
        variable_list = [index_variable]
        no_variable_list = list(range(n_variables))
        no_variable_list.remove(index_variable)
        data_var = [data[index_variable]]
        variable_probability = state_probability.get_probability(variable_list, data_var)
        data_no_var = data.copy()
        data_no_var.pop(index_variable)
        no_variable_probability = state_probability.get_probability(no_variable_list, data_no_var)
        local_o += (math.log2(variable_probability) - math.log2(no_variable_probability))
    return local_o


class LocalOHOI:

    def __init__(self, time_dimension, trial_dimension):
        self.bootstrap = False  # todo remove this argument and associated logic
        self.time_dimension = time_dimension
        self.trial_dimension = trial_dimension

    # todo add some check to see if baseline table has the same columns as data_table
    def exhaustive_local_o(self, data_table: pd.DataFrame):
        """Computes local o information and significance for all data states in data_table.
            Returns
            -------
                local_os : a dictionary of all local information associated with bootstrapped significance
                (local_o ci including 0 or not)
        """

        time_points = data_table[self.time_dimension].unique()
        trial_points = data_table[self.trial_dimension].unique()
        local_o_timepoints = []
        for time_point in tqdm(time_points):
            empirical_observations = data_table.loc[data_table[self.time_dimension] == time_point]
            empirical_observations = empirical_observations.iloc[:, 2:empirical_observations.shape[1]]
            states_probability = StatesProbabilities(empirical_observations)
            local_os = []
            for trial_point in trial_points:
                # get the row corresponding to time_point and trial_point
                data = data_table.loc[
                    (data_table[self.time_dimension] == time_point) & (data_table[self.trial_dimension] == trial_point)]
                data = data.iloc[:, 2:data.shape[1]].iloc[0]
                local_o = local_o_state(data, states_probability)
                local_os.append(local_o)
            local_o_timepoints.append(local_os)
        return local_o_timepoints
