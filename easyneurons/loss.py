import numpy as np

class CategoricalCrossEntropy:
  def forward(self, inputs, answers):
    self.inputs = inputs

    if len(answers.shape) == 1:
      confidences = inputs[range(len(inputs)), answers]

    elif lan(answers.shape) == 2:
      confidences = np.sum(inputs * answers, axis=1)

    confidences = np.clip(confidences, 1e-7, 1-1e-7)
    loss = -np.log(confidences)

    self.output = np.mean(loss)

class MeanSquaredError:
  def forward(self, inputs, answers):
    self.inputs = inputs
    
    self.output = np.mean((inputs - answers)**2, axis=-1)
