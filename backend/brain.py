import numpy as np


class NeuralNetwork:
    def __init__(self):
        # Initialise placeholders
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None

    def load_weights(self, path):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        e = np.exp(z - np.max(z))
        return e / e.sum(axis=0)

    def forward_pass(self, input_pixels):
        a0 = np.array(input_pixels).reshape(784, 1)

        # Layer 1
        z1 = np.dot(self.W1, a0) + self.b1
        a1 = self.sigmoid(z1)

        # Layer 2
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.softmax(z2)

        return {
            "input_layer": a0.flatten().tolist(),
            "hidden_layer": a1.flatten().tolist(),
            "output_layer": a2.flatten().tolist(),
            "prediction": int(np.argmax(a2))
        }