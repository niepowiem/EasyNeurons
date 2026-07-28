import numpy as np
import matplotlib.pyplot as plt
from easyneurons.metrics import *
from matplotlib.colors import ListedColormap

from easyneurons.general import Model, Dataset
from easyneurons.layer import NLayer
from easyneurons.activation import ReLU, Softmax
from easyneurons.loss import CategoricalCrossEntropy
from easyneurons.optimizer import *
from easyneurons.tokenization import *

class ClassificationModel:
    def __init__(self, inputs:np.ndarray, answers:np.ndarray, size: str='medium'):
        if size not in ("small", "medium", "big"):
            raise ValueError("Value \'size\' must be one of: small, medium, big")
        else:
            if len(answers.shape) == 2:
                n_labels = answers.shape[1]

                answers = np.argmax(answers, axis=1)
            else:
                n_labels = answers.max() + 1
            
            if size == "small":
                elements = [
                    NLayer(inputs.shape[1], 16),
                    ReLU(),
                    NLayer(16, 32),
                    ReLU(),
                    NLayer(32, 32),
                    ReLU(),
                    NLayer(32, n_labels),
                    Softmax(),
                ]
            elif size == "medium":
                elements = [
                    NLayer(inputs.shape[1], 64),
                    ReLU(),
                    NLayer(64, 128),
                    ReLU(),
                    NLayer(128, 128),
                    ReLU(),
                    NLayer(128, 128),
                    ReLU(),
                    NLayer(128, 128),
                    ReLU(),
                    NLayer(128, n_labels),
                    Softmax(),
                ]
            elif size == "big":
                elements = [
                    NLayer(inputs.shape[1], 128),
                    ReLU(),
                    NLayer(128, 512),
                    ReLU(),
                    NLayer(512, 512),
                    ReLU(),
                    NLayer(512, 512),
                    ReLU(),
                    NLayer(512, 512),
                    ReLU(),
                    NLayer(512, 512),
                    ReLU(),
                    NLayer(512, 512),
                    ReLU(),
                    NLayer(512, n_labels),
                    Softmax(),
                ]

            self.model = Model(elements=elements, loss=CategoricalCrossentropy())
            self.dataset = Dataset(inputs, answers, shuffle=True)
            self.optimizer = SGD(model=model, learning_rate=0.6, decay=0.001, momentum=0.9)
            
    def train(self):
        self.tracker = Tracker()
        self.metrics = CategoricalClassificationMetrics(accuracy=True,
                                                        precision=True,
                                                        recall=True,
                                                        f1=True,
                                                        mcc=True)
        
        self.optimizer.train(dataset=self.dataset,
                             epochs=1000,
                             tracker=self.tracker,
                             metrics=self.metrics)
        
    def predict(self, inputs: np.array) -> np.ndarray:
        return self.model.inference(inputs=inputs)
