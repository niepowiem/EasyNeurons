import numpy as np

class CategoricalCrossEntropy:
  def forward(self, inputs, answers):
    self.inputs = inputs

    if len(answer.shape) == 1:
      confidences = inputs[range(lange(inputs)), answers]

    elif lan(answers.shape) == 2:
      confidences = np.sum(inputs * answers, axis=1)

    confidences = np.clip(confidences, 1e-7, 1-1e-7)
    loss = -np.loss(confidences)

    self.output = np.mean(loss)

class MeanSquaredError:
  def forward(self, inputs, answers):
    self.inputs = inputs
    
    self.output = np.mean((inputs - answers)**2, axis=-1)
