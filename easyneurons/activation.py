import numpy as np
from easyneurons.general import NeuralElement

class ReLU(NeuralElement):
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
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

        return self

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        TBC
        :param dvalues:
        :return:
        """

        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

        return self

class LeakyReLU(NeuralElement):
    parameters = ("alpha",)

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        TBC
        :param inputs:
        :return:
        """

        self.inputs = inputs
        self.outputs = np.where(inputs > 0, inputs, self.alpha * inputs)

        return self

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        TBC
        :param dvalues:
        :return:
        """

        self.dinputs = dvalues.copy()

        # Where <0, gradient * alpha
        self.dinputs[self.inputs <= 0] *= self.alpha

        return self

class Softmax(NeuralElement):
    """
    ENG:
    This function is mapping results of output layer to probability distribution (only one answer). It's ment to be used only for output layer. For more:

    PL:
    Ta funkcja zwraca rozkład procentowy z wyników warstwy wyjściowej (tylko jedna odpowiedź). Należy ją stosować tylko na warstwie wyjściowej: Wyjaśnienie:
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Using formula: e^input of inputs / Σj e^inputs
        :param inputs: Matrix of neuronal output
        :return:
        """

        self.inputs = inputs

        suma = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = suma / np.sum(suma, axis=1, keepdims=True)

        return self

    def backward(self, dinputs: np.ndarray) -> np.ndarray:
        self.dinputs = np.empty_like(dinputs)

        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dinputs)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

        return self

class Sigmoid(NeuralElement):
    """
    ENG:
    This function is mapping results of output layer to probability distribution (multiple answers). For more watch:

    PL:
    Ta funkcja zwraca rozkład procentowy z wyników warstwy wyjściowej (wiele odpowiedzi). Wyjaśnienie:
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Using formula: 1 / (1 + e^(-input of inputs))
        :param inputs: Matrix of neuronal output
        :return:
        """
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))

        return self

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        self.dinputs = dvalues * (1 - self.outputs) * self.outputs

        return self

class ELU(NeuralElement):
    parameters = ("alpha",)

    def __init__(self, alpha:float=1):
        self.alpha = alpha

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = self.alpha * np.expm1(np.minimum(inputs, 0.0))
        # Minimum, ponieważ jeżeli policzymy ELU dla dużych dodatnich liczb, wartość może się przepełnić

        self.outputs = np.where(inputs > 0, inputs, self.outputs)

        return self

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        self.dinputs = dvalues * np.where(self.inputs > 0, 1.0,
                                          self.alpha * np.exp(
                                              np.minimum(self.inputs, 0.0)
                                            )
                                          )
        return self

class PReLU(NeuralElement):
    parameters = ("alpha", "multichannel")
    trainable = ("alpha",)

    def __init__(self, alpha:float=0.01, multichannel:bool=False):
        self.multichannel = multichannel
        self.alpha = alpha

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs

        if self.multichannel and not isinstance(self.alpha, np.ndarray):
            self.alpha = self.alpha * np.ones(inputs.shape[1])

        self.outputs = np.where(inputs < 0, self.alpha * inputs, inputs)

        return self

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        self.dinputs = dvalues * np.where(self.inputs > 0, 1.0, self.alpha)
        self.dalpha = dvalues * np.where(self.inputs > 0, 0.0, self.inputs)

        if self.multichannel:
            self.dalpha = np.sum(self.dalpha, axis=0)
        else:
            self.dalpha = np.sum(self.dalpha)

        return self

"""
To add:
0. PReLU
1. ELU
2. GELU
3. SELU
4. Linear
5. Binary
6. Swish
7. Mish
8. Sin
9. Tanh
10. Softsign
11. Softplus
12. Snake
13. GLU
14. ReLUClip
15. HardSigmoid
16. HardSwish
17. HardTanh
18. LogSoftmax
19. SiLU
20. CReLU
21. GumbelSoftmax
22. Softmin
23. Threshold
24. StarReLU
25. Gaussian
26. Sinc
27. ArcTanh
28. ISRU
29. BentIdentity
30. Maxout
31. SwiGLU
32. CELU
33. GeGLU
34. ReGLU
35. Abs
36. Cosine
37. SQNL
38. FReLU
"""
