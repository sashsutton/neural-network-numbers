import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        self.W3, self.b3 = None, None
        self.W4, self.b4 = None, None # Added 4th layer

    def load_weights(self, path):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W3, self.b3 = data["W3"], data["b3"]
        self.W4, self.b4 = data["W4"], data["b4"]

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        e = np.exp(z - np.max(z))
        return e / e.sum(axis=0)

    def forward_pass(self, input_pixels):
        if self.W1 is None:
            raise Exception("Weights not loaded!")

        a0 = np.array(input_pixels).reshape(784, 1)


        a1 = self.relu(np.dot(self.W1, a0) + self.b1)     # 512 neurons
        a2 = self.relu(np.dot(self.W2, a1) + self.b2)     # 256 neurons
        a3 = self.relu(np.dot(self.W3, a2) + self.b3)     # 128 neurons
        a4 = self.softmax(np.dot(self.W4, a3) + self.b4)  # 11 output neurons

        prediction_idx = int(np.argmax(a4))
        prediction_label = str(prediction_idx) if prediction_idx < 10 else "Not a Number"

        return {
            "input_layer": a0.flatten().tolist(),
            "hidden_layer1": a1.flatten().tolist(),
            "hidden_layer2": a2.flatten().tolist(),
            "hidden_layer3": a3.flatten().tolist(),
            "output_layer": a4.flatten().tolist(),
            "prediction": prediction_label,
            "confidence": float(np.max(a4))
        }