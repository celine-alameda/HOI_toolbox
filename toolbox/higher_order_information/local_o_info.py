"""Local O information computation.
    Work in progress.
 """
import concurrent.futures
import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

import pandas as pd
import scipy
from sklearn.utils import resample
from tqdm import tqdm

from toolbox.states_probabilities import StatesProbabilities, ProbabilityEstimator

n_workers = 8  # todo put into class


@dataclass
class CombinationWithDataRange:
    combination: tuple
    start: int
    to: int


def compute_probability(data_table, block: CombinationWithDataRange):
    data_to_compute = data_table.iloc[block.start:block.to, list(block.combination)]
    data_necessary = data_table.iloc[:, list(block.combination)]
    # transpose because the KDE is created with dimensions x datapoints
    # the data_necessary table is row x columns

    return scipy.stats.gaussian_kde(data_necessary.T).pdf(data_to_compute.T)


def compute_blocks(n_variables, n_datapoints) -> list:
    blocks = []
    combinations = [list(range(n_variables))]
    for index_variable in range(n_variables):
        variable_list = [index_variable]
        no_variable_list = list(range(n_variables))
        no_variable_list.remove(index_variable)
        combinations.append(variable_list)
        combinations.append(no_variable_list)
    # now we have all combinations
    # split them between workers
    full_worker_batches = len(combinations) // n_workers
    n_rest = len(combinations) % n_workers  # number of combinations left after feeding equally all workers
    if n_rest == 0 or n_rest > n_workers / 2:
        # in that case, just feed the rest of the combinations to the workers
        limit = len(combinations)
    else:
        limit = full_worker_batches * n_workers
    for i in range(limit):
        blocks.append(CombinationWithDataRange(tuple(combinations[i]), 0, n_datapoints))
    # split the rest of the combinations between the workers if needed
    if limit != len(combinations):
        # some combinations remain, split them between workers
        each_combination_split_into = n_workers // n_rest
        for i in range(limit, len(combinations)):
            combination = combinations[i]
            end = 0
            for j in range(each_combination_split_into):
                start = end
                if j == each_combination_split_into - 1:
                    end = n_datapoints
                else:
                    end = (j + 1) * (n_datapoints // each_combination_split_into)
                blocks.append(CombinationWithDataRange(tuple(combination), start, end))
    return blocks


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
        blocks = compute_blocks(n_variables, n_datapoints)
        probability_dict = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(compute_probability, data_table, block): block for block
                       in blocks}
            for future in concurrent.futures.as_completed(futures):
                block = futures[future]
                try:
                    probabilities = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (block, exc))
                else:
                    if block.combination not in probability_dict.keys():
                        probs = [0 for _ in range(n_datapoints)]
                        probability_dict[block.combination] = probs
                    probs = probability_dict[block.combination]
                    for i in range(len(probabilities)):
                        probs[i + block.start] = probabilities[i]
        local_os = []
        print("Probabilities Done.")
        print("Computing entropies from probabilities...")
        for i in tqdm(range(n_datapoints)):
            local_os.append(local_o_of_index(n_variables, i, probability_dict))
        return local_os
