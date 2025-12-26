import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 1. Load Data
print("Loading MNIST data...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data / 255.0  # Normalise pixels to [0, 1]
y = mnist.target.astype(int)

# Convert labels to "One-Hot"
y_one_hot = np.zeros((y.size, 10))
y_one_hot[np.arange(y.size), y] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2)

# 2. Training Parameters
input_size = 784
hidden_size = 16
output_size = 10
learning_rate = 0.1
epochs = 10

# Initialise Weights
W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): return sigmoid(z) * (1 - sigmoid(z))
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum(axis=0)

# 3. Training Loop
print("Starting Training...")
for epoch in range(epochs):
    for i in range(len(X_train)):
        # Forward Pass
        a0 = X_train[i].reshape(784, 1)
        z1 = np.dot(W1, a0) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = softmax(z2)

        # Backward Pass
        dz2 = a2 - y_train[i].reshape(10, 1) # Error at output
        dW2 = np.dot(dz2, a1.T)
        db2 = dz2

        dz1 = np.dot(W2.T, dz2) * sigmoid_deriv(z1) # Error at hidden
        dW1 = np.dot(dz1, a0.T)
        db1 = dz1

        # Update Weights
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    print(f"Epoch {epoch+1} complete")

# 4. Save the "Brain"
np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Saved weights to weights.npz!")