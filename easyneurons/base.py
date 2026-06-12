import os
import pickle
import numpy as np
from datetime import datetime

class NeuralElement:
    def get_parameters(self, copy: bool = True) -> dict:
        return None

    def set_parameters(self, params: dict):
        pass

class Loss(NeuralElement):
    pass

class Model:
    def __init__(self, elements: list(NeuralElement), loss: Loss):
        self.elements = elements
        self.loss = loss

    def forward(self, inputs: np.ndarray, answers: np.ndarray = None) -> dict:
        if not self.elements:
            raise ValueError("This model has no elements specified!")

        if inputs is None:
            raise ValueError("No inputs passed! Aborting...")

        if answers is None:
            print("No answers passed! Performing inference instead...")
            return self.inference(inputs)

        self.elements[0].forward(inputs)

        for element_id in range(1, len(self.elements)):
            self.elements[element_id].forward(
                self.elements[element_id-1].outputs
            )

        self.loss.forward(self.elements[-1].outputs, answers)

        return {"output": self.elements[-1].outputs, "loss": self.loss.outputs}

    def backward(self):
        if not self.elements:
            raise ValueError("This model has no elements specified!")
            return False

        if not self.loss:
            raise ValueError("This model has no loss specified!")

        self.elements[-1].backward(self.loss.backward())

        # -2, and not -1 because if array has 6 elements after -1 we get 5,
        # so we need -2 to get 4
        for element_id in range(len(self.elements) - 2, -1, -1):
            self.elements[element_id].backward(self.elements[element_id + 1].dinputs)

    def inference(self, inputs: np.ndarray) -> np.ndarray:
        print("Inference started...")

        self.elements[0].forward(inputs)

        for element_id in range(1, len(self.elements)):
            self.elements[element_id].forward(
                self.elements[element_id - 1].outputs
            )

        return self.elements[-1].outputs

    def __get__(self, instance, owner):
        if not self.elements:
            raise ValueError("This model has no elements specified!")

        return self.elements

    def save(self, name: str = None, path: str = None, save_loss: bool = False):

        model_data = {
            "architecture": [],
            "parameters": [],
        }

        for element in self.elements:
            model_data["architecture"].append(element.__class__.__name__)
            model_data["parameters"].append(element.get_parameters(False))

        if self.loss and save_loss:
            model_data["architecture"].append(self.loss.__class__.__name__)
            model_data["parameters"].append(self.loss.get_parameters(False))

        if not name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"model_{timestamp}"

        if not path:
            path = "models"

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{name}.pkl")

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, name: str = None, path: str = 'models'):

        if not name:
            raise ValueError("Model name not specified!")

        filepath = os.path.join(path, f"{name}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        architecture = model_data["architecture"]
        parameters = model_data["parameters"]

        model = cls.__new__(cls)
        model.elements = []
        model.loss = None

        class_registry = _build_registry(NeuralElement)

        for type, parameters in zip(architecture, parameters):
            class_object = class_registry[type]
            element = class_object.__new__(class_object)

            if parameters:
                element.set_parameters(parameters)

            if isinstance(element, Loss):
                model.loss = element
            else:
                model.elements.append(element)

        return model

class Dataset:
    def __init__(self, inputs: np.ndarray, answers: np.ndarray, batch_size: int = None, shuffle:bool=True, seed:int=None):
        if len(inputs) != len(answers):
            raise ValueError("The number of inputs and answers must match!")

        self.inputs = inputs
        self.answers = answers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) 

    def __iter__(self):
        n_samples = len(self.inputs)
        indices = self.rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)

        if not self.batch_size:
            yield self.inputs[indices], self.answers[indices]
            return

        for start in range(0, n_samples, self.batch_size):
            batch = indices[start:start + self.batch_size]
            yield self.inputs[batch], self.answers[batch]

    def __len__(self):
        return len(self.inputs)

def _build_registry(base_cls: __subclasses__) -> dict:
    registry = {}

    for sub in base_cls.__subclasses__():
        registry[sub.__name__] = sub
        registry.update(_build_registry(sub))

    return registry
