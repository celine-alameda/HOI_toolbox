"""state_probability.py

compute and retrieve probabilities for continuous, possibly polynomial, distributions"""

import pandas as pd
import scipy.stats

# fit an empirical cdf to a bimodal dataset
from matplotlib import pyplot


class StateProbability:
    _all_probability_density_functions = {}

    def __init__(self, data_table):
        """initializes the StateProbability object
        Parameters
        ----------
        data_table : pd.DataFrame
            the array with all variables and observations"""
        self._data_table = data_table
        self._n_variables = data_table.shape[1]
        self._n_samples = data_table.shape[0]

    def get_probability(self, which_variables, values) -> float:
        """computes the probability of a given state.
        Parameters
        ----------
        which_variables : list
            index of the variables in the joint distribution
        values : list
            values of the variables"""
        which_variables.sort()
        if tuple(which_variables) not in self._all_probability_density_functions.keys():
            pdf = self._create_pdf(which_variables)
            self._all_probability_density_functions[tuple(which_variables)] = pdf
            ### right thing ?? TODO
            pdf.pdf(values)
            print(pdf.pdf(values))

        # use it now
        return 1.0

    def _create_pdf(self, which_variables):
        data_necessary = self._data_table.iloc[:, which_variables]
        return scipy.stats.gaussian_kde(data_necessary)


