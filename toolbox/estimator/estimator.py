from abc import abstractmethod


class Estimator:

    def __init__(self):
        self.type = "undefined"

    @abstractmethod
    def estimate_entropy(self, x):
        pass

    @abstractmethod
    def estimate_cmi(self, y, x0, y0):
        pass
