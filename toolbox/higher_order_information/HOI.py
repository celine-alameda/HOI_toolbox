from toolbox.estimator.gcmi import GCMIEstimator
from toolbox.estimator.lin_est import LinearEstimator


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
