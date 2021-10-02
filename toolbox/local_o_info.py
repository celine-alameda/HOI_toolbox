"""Local O information computation.
    Work in progress.
 """
import math
import sys
import numpy as np
import itertools

import pandas as pd
from tqdm.auto import tqdm
from toolbox.gcmi import copnorm, ent_g
from toolbox.lin_est import lin_ent
from toolbox.utils import bootci, CombinationsManager, ncr
from toolbox.state_probability import StateProbability


def exhaustive_local_o(data_table: pd.DataFrame):
    """computes local o information for all data states in data_table."""
    n_variables = data_table.shape[1]
    n_samples = data_table.shape[0]
    local_os = []
    state_probability = StateProbability(data_table)
    # transpose data table so that data_table[i] returns data for trial i
    for index_sample in range(n_samples):
        data = data_table.iloc[index_sample, :]
        probability = state_probability.get_probability(list(range(n_variables)), data)
        local_o = (n_variables - 2) * entropy(probability)
        for index_variable in range(n_variables):
            variable_list = []
            no_variable_list = []
            for j in range(n_variables):
                if j == index_variable:
                    variable_list.append(j)
                else:
                    no_variable_list.append(j)
            data_var = [data[index_variable]]
            variable_probability = state_probability.get_probability(variable_list, data_var)
            data_no_var = data.copy()
            data_no_var.pop(index_variable)
            no_variable_probability = state_probability.get_probability(no_variable_list, data_no_var)
            local_o += (entropy(variable_probability) - entropy(no_variable_probability))
        local_os.append(local_o)
    return local_os


def entropy(probability):
    return math.log2(probability)
