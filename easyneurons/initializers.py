from abc import ABC, abstractmethod

from math import sqrt
import numpy as np

class Initialization(ABC):

    @abstractmethod
    def initialize(self, n_inputs: int, n_outputs: int) -> np.ndarray:
        pass

class StaticInitialization(Initialization):
    def __init__(self, number: float = 0.0):
        self.number = number

    def initialize(self, n_inputs: int, n_outputs: int) -> np.ndarray:
        return self.number * np.ones((n_inputs, n_outputs))

class RandomInitialization(Initialization):
    def __init__(self, mode: str = "normal", multiplier: float = 1.0, alpha:float|int = 1, seed: int = None):
        if mode not in ('uniform', 'gaussian', 'normal'):
            raise ValueError("Value of 'mode' must be one of 'uniform', 'gaussian', 'normal'")

        self.mode = mode
        self.multiplier = multiplier
        self.seed = seed
        self.alpha = alpha

    def initialize(self, n_inputs: int, n_outputs: int) -> np.ndarray:
        if self.mode == "uniform":
            return self._uniform(n_inputs, n_outputs, multiplier=self.multiplier, seed=self.seed, alpha=self.alpha)
        return self._normal(n_inputs, n_outputs, multiplier=self.multiplier, seed=self.seed)

    @staticmethod
    def _uniform(n_inputs: int, n_outputs: int, multiplier: float = 1.0, seed: int = None, alpha: float = 1.0) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        return multiplier * rng.uniform(-alpha, alpha, (n_inputs, n_outputs))

    @staticmethod
    def _normal(n_inputs: int, n_outputs: int, multiplier: float = 1.0, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed=seed)
        return multiplier * rng.standard_normal((n_inputs, n_outputs))

class XavierGlorotInitialization(RandomInitialization):
    def __init__(self, mode: str = "normal", seed: int = None):
        super().__init__(mode=mode, seed=seed)

    def initialize(self, n_inputs: int, n_outputs: int) -> np.ndarray:
        if self.mode == "uniform":
            alpha = sqrt(6 / (n_inputs + n_outputs))
            return self._uniform(n_inputs, n_outputs, seed=self.seed, alpha=alpha)

        std = sqrt(2 / (n_inputs + n_outputs))
        return self._normal(n_inputs, n_outputs, multiplier=std, seed=self.seed)

class HeKaimingInitialization(RandomInitialization):
    def __init__(self, mode: str = "normal", seed: int = None):
        super().__init__(mode=mode, seed=seed)

    def initialize(self, n_inputs: int, n_outputs: int) -> np.ndarray:
        if self.mode == "uniform":
            alpha = sqrt(6 / n_inputs)
            return self._uniform(n_inputs, n_outputs, seed=self.seed, alpha=alpha)

        std = sqrt(2 / n_inputs)
        return self._normal(n_inputs, n_outputs, multiplier=std, seed=self.seed)

class LeCunInitialization(RandomInitialization):
    def __init__(self, mode: str = "normal", seed: int = None):
        super().__init__(mode=mode, seed=seed)

    def initialize(self, n_inputs: int, n_outputs: int) -> np.ndarray:
        if self.mode == "uniform":
            alpha = sqrt(3 / n_inputs)
            return self._uniform(n_inputs, n_outputs, seed=self.seed, alpha=alpha)

        std = sqrt(1 / n_inputs)
        return self._normal(n_inputs, n_outputs, multiplier=std, seed=self.seed)

# class OrthogonalInitialization(Initialization):
#     def __init__(self, seed: int = None):
#         pass
#
#     def initialize(self, n_inputs: int, n_outputs: int) -> np.ndarray:
#         pass
