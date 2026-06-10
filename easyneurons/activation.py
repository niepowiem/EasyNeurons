import numpy as np

class ReLU:
    """
       Rectified Linear Unit (ReLU) activation function
       ------------------------------------------------

       This class introduces non-linearity by zeroing out negative
       values while leaving positive values unchanged.

       Mathematical formula
       --------------------
       f(x) = { x > 0: x, x < 0: 0 }

       Good For
       --------
       TBC

       Performance
       -----------
       TBC

       Troubleshooting
       ---------------
       **1. Dying ReLU problem:** Occurs when neurons persist in returning zero,
       thus resulting in zero gradient, which leads to neuron never activating
       and being unable to learn.

       Fixing the dying ReLU problem is essential becuse if many neurons die,
       the neural network's ability to learn complex data drops. It can slow down
       training and reduce performance.

       **Diagnosis:** The learning rate stays unchanged, even after couple of epochs

       **Causes**: This problem might be casued by:

       **a) Weight Initialization:** If weights are set in a way that makes the neurons outputs
       mainly negative values, after passing through the activation functions, these
       neurons may become permatetly inactive.

       **b) LARGE negative biases:** Even when the weight is positive, a large negative
       bias, after being added to w * x, can lead to negative output. Despite having
       a valid input and positive weight, the large negative bias may lead to the
       neuron outputting values below zero. If such biases persist across many neurons,
       they can cause these neurons to become inactive.

       **c) Too HIGH Learning Rate:** When learning rate is too high it can update parameters
       with large negative values. For example:\n
       dinput = 10\n
       w (weight) = 0.5\n
       lr (learning rate) = 1\n
       w_new = w - lr * dinput = 0.5 - 1 * 10 = -9.5

       **Solutions:** One of these solutions may help with the issue

       **a) Using LeakyReLU**, which allows non-zero gradient for negative values,
       ensuring the neurons maintain some level of activity, even when the input
       is negative.

       **b) Exploring other activation functions** such as: PReLU, ELU

       **c) Lower learning rates** can prevent weights from becoming excessively negative,
       after updates, which reduces the risk of neurons returning negative values

       **d) Proper initialization** can ensure, the neurons start
       with values more likely to keep them active in the early stages of learning

       **e) Batch Normalization** TBC

       **2. Unbounded Output:** Because ReLU is unbounded on the positive side, this
       can lead to gradients growing exponentially, reaching astronomical values
       thus destabilizing the model. Common issue in RNN.

       **Diagnosis:**
            a) Loss function suddenly jumps to absurdly high values\n
            b) During training, loss function returns 'NaN'\n
            c) Weights are 'inf'

       **Solutions:**

       **a) Gradient Clipping** is used for exploding gradients problem. This method
       clips the graients to be in a certain threshold for example to be between -4 and 4.
       This method has its drawback, because by clipping the gradient, the vector direction
       may change. To maintain the direction of the vector, we can **Clip By Norm**, which
       means, instead of only clipping values outside of the range, we are lowering all of
       our gradient values to be in betweeen -1 and 1. The drawback of this method is that
       some of the vales may become very small.

       **b) Proper initialization** TBC

       **c) Batch Normalization** TBC

       **3. Noisy Gradients:** 

       Video Explanation
       -----------
       **ENG:** ccc\n
       **PL:** ccc

       Attributes
       ----------
       input : np.ndarray
           The input passed to the forward method.
       output : np.ndarray
           The result after applying the activation function.
       """
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        TBC
        :param dvalues:
        :return:
        """
        self.input = inputs
        self.output = np.maximum(0, inputs)

        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        TBC
        :param dvalues:
        :return:
        """

        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative
        self.dvalues[self.inputs <= 0] = 0

        return dvalues

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        TBC
        :param inputs:
        :return:
        """

        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)

        return self.output

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        TBC
        :param dvalues:
        :return:
        """

        self.dinputs = dvalues.copy()

        # Where <0, gradient * alpha
        self.dinputs[self.inputs <= 0] *= self.alpha

        return self.dinputs

class PReLU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class ELU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class GELU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class SELU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Softmax:
    """
    ENG:
    This function is mapping results of output layer to probability distribution (only one answer). It's ment to be used only for output layer. For more:

    PL:
    Ta funkcja zwraca rozkład procentowy z wyników warstwy wyjściowej (tylko jedna odpowiedź). Należy ją stosować tylko na warstwie wyjściowej: Wyjaśnienie:
    """

    def forward(self, inputs):
        """
        Using formula: e^input of inputs / Σj e^inputs
        :param inputs: Matrix of neuronal output
        :return:
        """

        self.inputs = inputs

        suma = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = suma / np.sum(suma, axis=1, keepdims=True)

    def backward(self, dinputs):
        self.dvalues = np.empty_like(dinputs)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dinputs)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dvalues[index] = np.dot(jacobian_matrix, single_dvalues)

class Sigmoid:
    """
    ENG:
    This function is mapping results of output layer to probability distribution (multiple answers). For more watch:

    PL:
    Ta funkcja zwraca rozkład procentowy z wyników warstwy wyjściowej (wiele odpowiedzi). Wyjaśnienie:
    """

    def forward(self, inputs):
        """
        Using formula: 1 / (1 + e^(-input of inputs))
        :param inputs: Matrix of neuronal output
        :return:
        """
        self.input = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

class Linear:
    def forward(self, inputs):
        self.input = inputs

    def backward(self, dinputs):
        self.dvalues = np.ones(dinputs.shape)

class Binary:
    def forward(self, inputs):
        self.output = np.heaviside(inputs, 1)

class Swish:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Mish:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Sin:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Tanh:
    def forward(self, inputs):
        self.input = inputs
        self.output = np.tanh(input)

class Softsign:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Softplus:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Snake:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class GLU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class ReLUClip:
    def forward(self, inputs, max=6):
        self.input = inputs

class HardSigmoid:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class HardSwish:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class HardTanh:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class LogSoftmax:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class SiLU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class CReLU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class GumbelSoftmax:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Softmin:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Threshold:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class StarReLU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Gaussian:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Sinc:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class ArcTanh:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class ISRU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class BentIdentity:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Maxout:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class SwiGLU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class CELU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class GeGLU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class ReGLU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Abs:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class Cosine:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class SQNL:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs

class FReLU:
    def forward(self, inputs, alpha=0.01):
        self.input = inputs
