import os
import pickle
import numpy as np
from datetime import datetime

class NeuralElement:
    parameters: tuple[str, ...] = ()
    trainable: tuple[str, ...] = ()

    def get_parameters(self) -> dict:
        return {key: getattr(self, key) for key in self.parameters}

    def get_trainable(self) -> dict:
        return {key: getattr(self, key) for key in self.trainable}

    def set_parameters(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)

    def __gt__(self, other):
        other.forward(self.outputs)
        return other

    def __rshift__(self, other):
        other.backward(self.dinputs)
        return other

class Loss(NeuralElement):
    pass

class Model:
    def __init__(self, elements: list[NeuralElement], loss: Loss):
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
            model_data["parameters"].append(element.get_parameters())

        if self.loss and save_loss:
            model_data["architecture"].append(self.loss.__class__.__name__)
            model_data["parameters"].append(self.loss.get_parameters())

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

        def split(self, split: float = 0.5) -> tuple[Dataset, Dataset]:
        if split > 1 or split < 0:
            raise ValueError("split must be between 0 and 1")

        middle_position = math.ceil(self.len(self.inputs) * split)

        return (Dataset(self.inputs[0:middle_position],
                             self.answers[0:middle_position],
                             batch_size=self.batch_size,
                             shuffle=self.shuffle),
                Dataset(self.inputs[middle_position:],
                        self.answers[middle_position:],
                        batch_size=self.batch_size,
                        shuffle=self.shuffle))
    
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

class Tracker:
    def __init__(self, every: int = 1, print: bool=True):
        if every < 1:
            raise ValueError("every must be greater or equal to 1")

        self.data = {}
        self.every = every
        self.calls = 0

        self.print = print

    def log(self, data: dict):
        self.calls += 1

        if self.calls % self.every == 0:
            for key, value in data.items():
                self.data.setdefault(key, []).append(value)

    def clear(self):
        self.data = {}

    def __repr__(self):
        if not self.data:
            return "No data found!"

        return ", ".join([f"{key}: {value[-1]}" for key, value in self.data.items()])

def _build_registry(base_cls) -> dict:
    registry = {}

    for sub in base_cls.__subclasses__():
        registry[sub.__name__] = sub
        registry.update(_build_registry(sub))

    return registry
