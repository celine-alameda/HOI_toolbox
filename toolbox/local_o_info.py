"""Local O information computation.
    Work in progress.
 """
import math
import sys
import numpy as np
import itertools

import pandas as pd
from sklearn.utils import resample
from tqdm.auto import tqdm
from toolbox.gcmi import copnorm, ent_g
from toolbox.lin_est import lin_ent
from toolbox.utils import bootci, CombinationsManager, ncr
from toolbox.states_probabilities import StatesProbabilities


def local_o_state(data, state_probability):
    n_variables = data.shape[0]
    system_probability = state_probability.get_probability(list(range(n_variables)), data)
    local_o = (n_variables - 2) * math.log2(system_probability)
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
        local_o += (math.log2(variable_probability) - math.log2(no_variable_probability))
    return local_o


class LocalOHOI:

    def __init__(self, bootstrap):
        self.bootstrap = bootstrap

    def exhaustive_local_o(self, data_table: pd.DataFrame):
        """Computes local o information and significance for all data states in data_table.
            Returns
            -------
                local_os : a dictionary of all local information associated with bootstrapped significance
                (local_o ci including 0 or not)
        """

        alpha = 0.05
        local_os = []
        significances = []
        lower_cis = []
        upper_cis = []
        n_rows = data_table.shape[0]
        states_probability = StatesProbabilities(data_table)
        for index_sample, row in data_table.iterrows():
            local_o = local_o_state(row, states_probability)
            local_os.append(local_o)
        # significances
        # local o bootstrap
        if self.bootstrap:
            local_o_bootstraps = pd.DataFrame()
            for _ in range(100):
                local_o_bootstrap = {}
                # todo replace 100 by n_boot
                sample = resample(range(n_rows), n_samples=len(range(n_rows)))
                s_data = pd.DataFrame(data_table.iloc[sample, :])
                # new definition of probabilities based on the sampled data
                states_probability = StatesProbabilities(s_data)
                # for each s_data, evaluate the local o of each row
                for index_sample, row in data_table.iterrows():
                    local_o = local_o_state(row, states_probability)
                    local_o_bootstrap[index_sample] = local_o
                local_o_bootstraps = local_o_bootstraps.append(local_o_bootstrap, ignore_index=True)
            # each row of the original dataframe is a col in the local_o_bootstraps
            for index_sample in range(n_rows):
                boostraps_for_sample = local_o_bootstraps.iloc[:, index_sample]
                stats = boostraps_for_sample.values.tolist()
                print(stats)
                p = (alpha / 2.0) * 100
                lower = np.percentile(stats, p)
                lower_cis.append(lower)
                p = (1 - alpha / 2.0) * 100
                upper = np.percentile(stats, p)
                upper_cis.append(upper)
                sig = 0 if lower <= 0 <= upper else 1
                significances.append(sig)
        return_object = {"local_o": local_os, "significances": significances, "lower_ci": lower_cis,
                         "upper_ci": upper_cis} if self.bootstrap else {"local_o": local_os}
        # confidence intervals for each row
        return return_object
