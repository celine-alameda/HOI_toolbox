"""states_probabilities.py

compute and retrieve probabilities for continuous, possibly polynomial, distributions"""
from abc import abstractmethod

import pandas as pd
import scipy.stats


class ProbabilityEstimator:

    @abstractmethod
    def get_probability(self, which_variables, values) -> float:
        pass


class StatesProbabilities(ProbabilityEstimator):

    def __init__(self, empirical_observations: pd.DataFrame):
        """initializes the StateProbability object
        Parameters
        ----------
        empirical_observations : pd.DataFrame
            the array to use for probability computing"""

        self._data_table = empirical_observations
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
