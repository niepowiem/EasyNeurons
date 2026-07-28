import numpy as np
from easyneurons import layer
from easyneurons.general import Model, Dataset, Tracker
from easyneurons.metrics import Metrics
from easyneurons.layer import *
from easyneurons.layer import NLayer
from itertools import batched
from tqdm import tqdm

class Optimizer(ABC):
    parameters: tuple[str, ...] = ("epochs", "iterations", "learning_rate", "current_learning_rate", "decay")

    def __init__(self, model: Model, learning_rate: float = 1.0, decay: float = 0.):
        self.model = model

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay

        self.epochs = 0
        self.iterations = 0

    def pre_update_parameters(self):
        if self.decay:
            # L/(1+d*x)
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.epochs * self.decay))

    @abstractmethod
    def update_parameters(self, element: NeuralElement):
        raise NotImplementedError

    def train(self, dataset: Dataset, epochs: int, tracker: Tracker = None, metrics: Metrics = None, callback=None):
        for epoch in range(epochs):
            self.pre_update_parameters()
            self.iterations = 0

            # Loss for tacker
            local_loss = 0
            for inputs, answers in dataset:
                results = self.model.forward(inputs, answers)

                self.model.backward()
                for element in self.model.elements:
                    self.update_parameters(element)

                local_loss += results["loss"]

                if metrics:
                    metrics.add(results["output"], answers)

                if callback:
                    callback({
                        "results": results,
                        "answers": answers
                    })

                self.iterations += 1

            if tracker:
                tracker.log({"loss": local_loss / self.iterations} | self.get_parameters())

            if metrics:
                metrics.calculate(force=False)
                metrics.clear()

            self.post_update_parameters()

    def post_update_parameters(self):
        self.epochs += 1

    def get_parameters(self) -> dict:
        return {key: getattr(self, key) for key in self.parameters}

class SGD(Optimizer):
    parameters = Optimizer.parameters + ("momentum",)

    def __init__(self, model: Model, learning_rate:float | int=1.0, decay:float | int=0., momentum:float | int=0.):
        super().__init__(model, learning_rate, decay)
        self.momentum = momentum

    def update_parameters(self, element: NeuralElement):
        parameters = element.get_trainable()

        if not parameters:
            return

        updates = { }

        if self.momentum:
            # Inicjalizacja momentów jesli nie istnieją
            if not hasattr(element, '_momentums'):
                element._momentums = {key: np.zeros_like(value) for key, value in parameters.items()}

            for key, value in parameters.items():
                gradient = getattr(element, f"d{key}")

                element._momentums[key] = self.momentum * element._momentums[key] - self.current_learning_rate * gradient
                updates[key] = value + momentum_update

        else:
            for key, value in parameters.items():
                gradient = getattr(element, f"d{key}")

                updates[key] = value - self.current_learning_rate * gradient

        element.set_parameters(updates)

class AdaGrad(Optimizer):
    parameters = Optimizer.parameters + ("epsilon",)

    def __init__(self, model: Model, learning_rate: float | int = 1.0, decay: float | int = 0., epsilon: float = 1e-7):
        super().__init__(model, learning_rate, decay)
        self.epsilon = epsilon

    def update_parameters(self, element: NeuralElement):
        parameters = element.get_trainable()

        if not parameters:
            return

        updates = { }

        if not hasattr(element, '_cache'):
            element._cache = {key: np.zeros_like(value) for key, value in parameters.items()}

        for key, value in parameters.items():
            gradient = getattr(element, f"d{key}")

            element._cache[key] += gradient ** 2
            updates[key] = value - self.current_learning_rate * gradient / (np.sqrt(element._cache[key]) + self.epsilon)

        element.set_parameters(updates)

class RMSProp(Optimizer):
    parameters = Optimizer.parameters + ("epsilon", 'rho')

    def __init__(self, model: Model, learning_rate: float | int = 1.0, decay: float | int = 0., epsilon: float = 1e-7, rho: float = 0.9):
        super().__init__(model, learning_rate, decay)
        self.epsilon = epsilon
        self.rho = rho

    def update_parameters(self, element: NeuralElement):
        parameters = element.get_trainable()

        if not parameters:
            return

        updates = { }

        if not hasattr(element, '_cache'):
            element._cache = {key: np.zeros_like(value) for key, value in parameters.items()}

        for key, value in parameters.items():
            gradient = getattr(element, f"d{key}")

            element._cache[key] = self.rho * element._cache[key] + (1 - self.rho) * gradient ** 2
            updates[key] = value - self.current_learning_rate * gradient / (np.sqrt(element._cache[key]) + self.epsilon)

        element.set_parameters(updates)

class Adam(Optimizer):
    parameters = Optimizer.parameters + ("epsilon", "beta_1", 'beta_2')

    def __init__(self, model: Model, learning_rate: float | int = 1.0, decay: float | int = 0., epsilon: float = 1e-7, beta_1: float = 0.9, beta_2: float = 0.999):
        super().__init__(model, learning_rate, decay)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_parameters(self, element: NeuralElement):
        parameters = element.get_trainable()

        if not parameters:
            return

        updates = { }

        if not hasattr(element, '_cache'):
            element._cache = {key: np.zeros_like(value) for key, value in parameters.items()}

        if not hasattr(element, '_momentums'):
            element._momentums = {key: np.zeros_like(value) for key, value in parameters.items()}

        for key, value in parameters.items():
            gradient = getattr(element, f"d{key}")

            element._momentums[key] = self.beta_1 * element._momentums[key] + (1 - self.beta_1) * gradient
            momentums_corrected = element._momentums[key] / (1 - self.beta_1 ** (self.iterations + 1))

            element._cache[key] = self.beta_2 * element._cache[key] + (1 - self.beta_2) * gradient ** 2
            cache_corrected = element._cache[key] / (1 - self.beta_2 ** (self.iterations + 1))

            updates[key] = value - self.current_learning_rate * momentums_corrected / (np.sqrt(cache_corrected) + self.epsilon)

        element.set_parameters(updates)

class AdamW(Optimizer):
    parameters = Optimizer.parameters + ("epsilon", "beta_1", 'beta_2')

    def __init__(self, model: Model, learning_rate: float | int = 1.0, decay: float | int = 0., epsilon: float = 1e-7, beta_1: float = 0.9, beta_2: float = 0.999):
        super().__init__(model, learning_rate, decay)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_parameters(self, element: NeuralElement):
        parameters = element.get_trainable()

        if not parameters:
            return

        updates = {}

        if not hasattr(element, '_cache'):
            element._cache = {key: np.zeros_like(value) for key, value in parameters.items()}

        if not hasattr(element, '_momentums'):
            element._momentums = {key: np.zeros_like(value) for key, value in parameters.items()}

        for key, value in parameters.items():
            gradient = getattr(element, f"d{key}")

            element._momentums[key] = self.beta_1 * element._momentums[key] + (1 - self.beta_1) * gradient
            momentums_corrected = element._momentums[key] / (1 - self.beta_1 ** (self.iterations + 1))

            element._cache[key] = self.beta_2 * element._cache[key] + (1 - self.beta_2) * gradient ** 2
            cache_corrected = element._cache[key] / (1 - self.beta_2 ** (self.iterations + 1))

            updates[key] = value - self.current_learning_rate * momentums_corrected / (np.sqrt(cache_corrected) + self.epsilon) - self.current_learning_rate * self.weight_decay * value

        element.set_parameters(updates)

class Muon(Optimizer):
    parameters = Optimizer.parameters + ("mode", "nestrov", "steps", "momentum", "epsilon")

    def __init__(self, model: Model, learning_rate: float | int = 1.0, decay: float | int = 0., mode: str="moonlight", nestrov: bool=True, steps: int = 5, momentum: float = 0.9, epsilon: float = 1e-7):
        mode = mode.lower()
        if mode not in ("moonlight", "kellerjordan", "naive", "mup"):
            raise ValueError("Value of 'mode' must be one of 'moonlight', 'kellerjordan', 'naive', 'MuP'")

        super().__init__(model, learning_rate, decay)
        self.mode = mode
        self.nestrov = nestrov
        self.steps = steps
        self.momentum = momentum
        self.epsilon = epsilon

    def update_parameters(self, element: NeuralElement):
        parameters = element.get_trainable()

        if not parameters:
            return

        updates = { }

        if not hasattr(element, '_momentums'):
            element._momentums = {key: np.zeros_like(value) for key, value in parameters.items()}

        for key, value in parameters.items():
            gradient = getattr(element, f"d{key}")

            momentum = self.momentum * element._momentums[key] + (1 - self.momentum) * gradient
            element._momentums[key] = momentum

            # Muon może operować tylko na macierzach, w innym przypadku stosuje się inny optymalizator
            is_matrix = np.ndim(value) == 2 and min(np.shape(value)) > 1
            if is_matrix:
                updates[key] = value - self.current_learning_rate * self._get_scale(*value.shape) * self._newton_schulz_iteration(G=momentum)

            else:
                updates[key] = value - self.current_learning_rate * momentum

        element.set_parameters(updates)

    def _newton_schulz_iteration(self, G: np.ndarray, a: float = 3.4445, b: float = -4.7750, c: float = 2.0315):
        X = G / (np.linalg.norm(G, 'fro') + self.epsilon)
        transpose = X.shape[0] > X.shape[1]

        if transpose:
            X = X.T

        for _ in range(self.steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        return X.T if transpose else X

    def _get_scale(self, n_inputs: int, n_outputs: int) -> float:
        if self.mode == "naive":
            return 1

        elif self.mode == "kellerjordan":
            return sqrt(max(1, n_outputs / n_inputs))

        elif self.mode == "mup":
            return sqrt(n_outputs / n_inputs)

        elif self.mode == "moonlight":
            return 0.2 * sqrt(max(n_outputs, n_inputs))

        else:
            raise ValueError("'mode' is not one of 'moonlight', 'kellerjordan', 'mup' or 'moon'")
