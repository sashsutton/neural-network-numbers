import numpy as np

class NeuralNetwork:
    def __init__(self):
        # 784 -> 16 -> 10 architecture
        self.w1 = np.random.randn(16, 784) * 0.01 #hidden weights
        self.b1 = np.zeros((16,1)) #hidden bias
        self.w2 = np.random.randn(10, 16) * 0.01 #output weights
        self.b2 = np.zeros((10,1)) #output bias

        def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))

        def softmax(self, z):
            exp_z = np.exp(z - np.max(z))
            return exp_z / exp_z.sum(axis=0)

        def forward_pass(self, input_pixels):
            """
            Runs the input through the network and returns ALL activations
            so we can see them in 3D.
            """

            # Input layer
            A0 = np.array(input_pixels).reshape(1, 784)

            # Layer 1: hidden
            Z1 = np.dot(self.w1, A0) + self.b1
            A1 = self.sigmoid(Z1)

            # Layer 2: Output
            Z2 = np.dot(self.w2, A1) + self.b2
            A2 = self.softmax(Z2)

            return{
                "input_layer": A0.flatten().tolist(), # To light up 3D pixels
                "hidden_layer": A1.flatten().tolist(), # To light up 3D hidden nodes
                "output_layer": A2.flatten().tolist(), # To show the final prediction
                "prediction": int(np.argmax(A2))
            }




