import numpy as np
from easyneurons import layer
from easyneurons.general import Model, Dataset, Tracker
from easyneurons.metrics import Metrics
from easyneurons.layer import *
from easyneurons.layer import NLayer
from itertools import batched
from tqdm import tqdm

class Optimizer:
    def get_parameters(self, copy=False):
        return None

class SGD(Optimizer):
    def __init__(self, model: Model, learning_rate:float | int=1.0, decay:float | int=0., momentum:float | int=0.):
        self.model = model

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate

        self.decay = decay
        self.momentum = momentum

        self.iterations = 0
        self.epochs = 0

    def get_parameters(self, copy: bool = True):
        if copy:
            return {
                "epochs": self.epochs.copy(),
                "iterations": self.iterations.copy(),
                "learning_rate": self.current_learning_rate.copy(),
                "decay": self.decay.copy(),
                "momentum": self.momentum.copy()
            }

        return {
            "epochs": self.epochs,
            "iterations": self.iterations,
            "learning_rate": self.current_learning_rate,
            "decay": self.decay,
            "momentum": self.momentum
        }

    def pre_update_parameters(self):
        if self.decay:
            # L/(1+d*x)
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.epochs * self.decay))

    def update_parameters(self, element: NeuralElement):
        parameters = element.get_parameters(copy=False)

        if not parameters:
            return

        updates = { }

        if self.momentum:
            # Inicjalizacja momentów jesli nie istnieją
            if not hasattr(element, '_momentums'):
                element._momentums = {key: np.zeros_like(value) for key, value in parameters.items()}

            for key, value in parameters.items():
                momentum_update = self.momentum * element._momentums[key] - self.current_learning_rate * getattr(element, f"d{key}")
                element._momentums[key] = momentum_update
                updates[key] = value + momentum_update

        else:
            for key, value in parameters.items():
                updates[key] = value - self.current_learning_rate * getattr(element, f"d{key}")

        element.set_parameters(updates)

    def post_update_parameters(self):
        self.epochs += 1

    def train(self, dataset: Dataset, epochs: int, tracker: Tracker=None, metrics: Metrics=None, callback=None):
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
                tracker.log({"loss": local_loss / self.iterations} | self.get_parameters(copy=False))

            if metrics:
                metrics.calculate(force=False)
                metrics.clear()

            self.post_update_parameters()
