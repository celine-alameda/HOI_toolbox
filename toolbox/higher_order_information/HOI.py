from abc import abstractmethod

from toolbox.estimator.estimator import Estimator
from toolbox.estimator.gcmi_estimator import GCMIEstimator
from toolbox.estimator.linear_estimator import LinearEstimator


class HOI:

    def __init__(self, config):
        self.config = config
        if config["estimator"] == 'lin_est':
            self.estimator: Estimator = LinearEstimator()
        elif config["estimator"] == 'gcmi':
            self.estimator: Estimator = GCMIEstimator()
        else:
            print("Please use estimator out of the following - 'lin_est' or 'gcmi'")
            exit(1)

    @abstractmethod
    def run(self, data_frame, config):
        pass
