import numpy as np

class ReLU:
    """
    ENG:
    This function is supposed to be put on top of neuronal outputs of hidden layers. This will add nonlinearity to the neural network. For more watch:

    PL:
    Ta funkcja ma być nakładana na wyniki neuronów ukrytych warstw. Wprowadzi to nieliniowość do sieci neuronowej. Wyjaśnienie:
    """

    def forward(self, inputs):
        """
        This function is setting any number of inputs that is less than 0 to 0

        :param inputs: Matrix of neuronal outputs
        :return:
        """

        self.output = np.maximum(0, inputs)

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

        suma = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = suma / np.sum(suma, axis=1, keepdims=True)

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

        self.output = 1 / (1 + np.exp(-inputs))

class Binary:
    def forward(self, inputs):
        self.output = np.heaviside(inputs, 1)
