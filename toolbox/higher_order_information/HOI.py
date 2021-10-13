from toolbox.estimator.gcmi_estimator import GCMIEstimator
from toolbox.estimator.linear_estimator import LinearEstimator


class HOI:

    def __init__(self, config):
        self.config = config
        if config["estimator"] == 'lin_est':
            self.estimator = LinearEstimator()
        elif config["estimator"] == 'gcmi':
            self.estimator = GCMIEstimator()
        else:
            print("Please use estimator out of the following - 'lin_est' or 'gcmi'")
            exit(1)
