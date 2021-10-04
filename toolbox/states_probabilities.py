"""states_probabilities.py

compute and retrieve probabilities for continuous, possibly polynomial, distributions"""
import math

import pandas as pd
import scipy.stats


class StatesProbabilities:


    def __init__(self, data_table: pd.DataFrame):
        """initializes the StateProbability object
        Parameters
        ----------
        data_table : pd.DataFrame
            the array with all variables and observations"""
        self._data_table = data_table
        self._n_variables = data_table.shape[1]
        self._n_samples = data_table.shape[0]
        self._all_probability_density_functions = {}

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
            kde = self._create_kde(which_variables)
            self._all_probability_density_functions[tuple(which_variables)] = kde
        else:
            kde = self._all_probability_density_functions[tuple(which_variables)]

        # use it now
        return kde.pdf(values)

    def _create_kde(self, which_variables):
        data_necessary = self._data_table.iloc[:, which_variables]
        return scipy.stats.gaussian_kde(data_necessary.T)
