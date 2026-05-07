import numpy as np

class NLayer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = 0.01 * np.random.randn(n_inputs,n_outputs)
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

input_ = [1, 2, 3]

# Oczekuje 3 wejść
# Zwraca 3 wyjścia
# w1 = [[w, w, w, w, w],
#       [w, w, w, w, w],
#       [w, w, w, w, w]]
warstwa1 = NLayer(3,5)

# Oczekuje 5 wejść
# Zwraca 3 wyjścia
# w2 = [[w, w, w],
#       [w, w, w],
#       [w, w, w],
#       [w, w, w],
#       [w, w, w]]
warstwa2 = NLayer(5,3)

# Oczekuje 3 wejść
# Zwraca 3 wyjścia
# w3 = [[w, w, w],
#       [w, w, w],
#       [w, w, w]]
warstwa3 = NLayer(3,3)

warstwa1.forward(input_)
warstwa2.forward(warstwa1.output)
warstwa3.forward(warstwa2.output)

print(warstwa3.output)
