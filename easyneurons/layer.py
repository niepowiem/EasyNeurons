import numpy as np

class NLayer:
    """
    ENG:
    This class is a simple neuronal layer. For more watch: https://www.youtube.com/watch?v=Q_qKSvRTkbk

    PL:
    Ta klasa to prosta warstwa neuronów. Wyjaśnienie: https://www.youtube.com/watch?v=W-tN-7qrv0k
    """

    def __init__(self, n_inputs, n_outputs):
        """
        :param n_inputs: Number of inputs this layer in requesting
        :param n_outputs: Number of outputs of this layer
        """

        self.weights = 0.01 * np.random.randn(n_inputs,n_outputs)
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs):
        """
        ENG:
        This function if performing neuronal calculations - dot product

        PL:
        Ta funkcja wykonuje obliczenia neuronowe - mnożenie macierzowe

        :param inputs: Inputs from the previous layer or activation function
        :return:
        """

        self.output = np.dot(inputs, self.weights) + self.biases