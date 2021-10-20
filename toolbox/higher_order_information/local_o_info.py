"""Local O information computation.
    Work in progress.
 """
import concurrent.futures
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
        chunked_data_points = create_n_chunks(n_data_points, 1000)
        local_os = [0 for _ in range(n_data_points)]
        n_data_computed = 0
        task_status = [False for _ in range(len(chunked_data_points))]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            future_chunked_local_os = {executor.submit(self._local_o_for_chunk, data_table, chunk): chunk for chunk in
                                       chunked_data_points}
            for future in concurrent.futures.as_completed(future_chunked_local_os):
                chunked_local_os = future_chunked_local_os[future]
                try:
                    data = future.result()
                    for index in data:
                        local_os[index] = data[index]
                    n_data_computed += len(data)
                    print("{} data computed".format(n_data_computed))
                except Exception as exc:
                    print('%r generated an exception: %s' % (chunked_local_os, exc))
                # else:
                #    print('%r page is %d bytes' % (chunked_local_os, len(data)))
        return local_os

    def _local_o_for_chunk(self, data_table: pd.DataFrame, chunk_of_indices):
        return_dict = {}
        for index in chunk_of_indices:
            state = data_table.iloc[index]
            local_o = local_o_of_state(state, self.probability_estimator)
            return_dict[index] = local_o
        return return_dict


def create_n_chunks(n_data, chunk_size):
    """ Creates chunks of chunk_size data.
    The data that did not fit, i.e n_data % chunk_size, will land in the last one.
    """
    chunks = []
    counter = 0
    while counter < n_data:
        if counter + chunk_size < n_data:
            chunks.append(list(range(counter, counter + chunk_size)))
        else:
            chunks.append(list(range(counter, n_data)))
        counter = counter + chunk_size
    return chunks
