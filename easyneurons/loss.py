import numpy as np
from .general import Loss

class CategoricalCrossEntropy(Loss):
    def forward(self, inputs: np.ndarray, answers: np.ndarray) -> float | int:
        self.answers = answers
        self.inputs = inputs

        if len(answers.shape) == 1:
            confidences = inputs[range(len(inputs)), answers]

        elif len(answers.shape) == 2:
            confidences = np.sum(inputs * answers, axis=1)

        confidences = np.clip(confidences, 1e-7, 1-1e-7)
        loss = -np.log(confidences)

        self.outputs = np.mean(loss)

    def backward(self) -> np.ndarray:
        samples = len(self.inputs)
        labels = len(self.inputs[0])

        if len(self.answers.shape) == 1:
            self.answers = np.eye(labels)[self.answers]

        self.dinputs = -self.answers / self.inputs
        self.dinputs = self.dinputs / samples

        return self.dinputs

class MeanSquaredError(Loss):
    def forward(self, inputs: np.ndarray, answers: np.ndarray):
        self.answers = answers
        self.inputs = inputs
    
        # self.outputs = np.mean((inputs - answers)**2, axis=-1)
        self.outputs = np.mean((inputs - answers)**2)

        return self.outputs

    def backward(self):
        samples = len(self.inputs)
        labels = len(self.inputs[0])

        # Gradient on values
        self.dinputs = -2 * (self.answers - self.inputs) / labels
        self.dinputs = self.dinputs / samples

        return self.dinputs

class BinaryCrossEntropy(Loss):
    def forward(self, inputs: np.ndarray, answers: np.ndarray) -> float | int:
        self.answers = answers
        self.inputs = inputs

        inputs_clipped = np.clip(inputs, 1e-7, 1 - 1e-7)

        sample_losses = -(answers * np.log(inputs_clipped) + (1 - answers) * np.log(1 - inputs_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        self.outputs = np.mean(sample_losses)

    def backward(self):
        samples = len(self.inputs)
        labels = len(self.inputs[0])

        clipped_dvalues = np.clip(self.inputs, 1e-7, 1 - 1e-7)

        self.dinputs = -(self.answers / clipped_dvalues - (1 - self.answers) / (1 - clipped_dvalues)) / labels
        self.dinputs = self.dinputs / samples

        return self.dinputs

class MeanAbsoluteError(Loss):
    def forward(self, inputs: np.ndarray, answers: np.ndarray):
        self.answers = answers
        self.inputs = inputs

        # self.outputs = np.mean(np.abs(answers - inputs), axis=-1)
        self.outputs = np.mean(np.abs(answers - inputs))

        return self.outputs

    def backward(self):
        samples = len(self.inputs)
        labels = len(self.inputs[0])

        # Calculate gradient
        self.dinputs = np.sign(self.answers - self.inputs) / labels
        self.dinputs = self.dinputs / samples

        return self.dinputs
