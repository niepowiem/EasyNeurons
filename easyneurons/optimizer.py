import numpy as np
from easyneurons import layer
from easyneurons.base import Model, Dataset
from easyneurons.layer import *
from easyneurons.layer import NLayer
from itertools import batched

class StochasticGradientDescent_depricated:
    def __init__(self, learning_rate=1.0, decay=0., momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate

        self.decay = decay
        self.momentum = momentum

        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            # L/(1+d*x)
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.iterations * self.decay))

            # L/((1+d)^x)
            # self.current_learning_rate /= 1 + self.decay

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

class SGD:
    def __init__(self, model: Model, learning_rate:float | int=1.0, decay:float | int=0., momentum:float | int=0.):
        self.model = model

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate

        self.decay = decay
        self.momentum = momentum

        self.iterations = 0
        self.epochs = 0

    def pre_update_parameters(self):
        if self.decay:
            # L/(1+d*x)
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.iterations * self.decay))

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
        self.iterations += 1

    def train(self, dataset: Dataset, epochs: int, callback=None):
        for epoch in range(epochs):
            self.pre_update_parameters()

            results = None
            answers = None

            for inputs, answers in dataset:
                results = self.model.forward(inputs, answers)
                self.model.backward()

                for element in self.model.elements:
                    self.update_parameters(element)

                self.post_update_parameters()

            if callback:
                callback(self, results, answers)

            self.epochs += 1
