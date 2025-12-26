import numpy as np
import tensorflow as tf # Used only for the dataset download

# 1. Load Data via Keras
print("Downloading MNIST data...")
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess Data
# Flatten 28x28 images into 784-length arrays and normalize to [0, 1]
X_train = X_train_raw.reshape(-1, 784) / 255.0
X_test = X_test_raw.reshape(-1, 784) / 255.0

# Convert labels to "One-Hot"
def one_hot(y):
    oh = np.zeros((y.size, 10))
    oh[np.arange(y.size), y] = 1
    return oh

y_train = one_hot(y_train_raw)
y_test = one_hot(y_test_raw)

# 3. Initialize Weights (784 -> 16 -> 10)
W1 = np.random.randn(16, 784) * 0.01
b1 = np.zeros((16, 1))
W2 = np.random.randn(10, 16) * 0.01
b2 = np.zeros((10, 1))

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): return sigmoid(z) * (1 - sigmoid(z))
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum(axis=0)

# 4. Training Loop
print("Starting Training (This may take a few minutes)...")
learning_rate = 0.1
epochs = 5 # Reduced epochs for faster initial test

for epoch in range(epochs):
    for i in range(len(X_train)):
        # Forward Pass
        a0 = X_train[i].reshape(784, 1)
        z1 = np.dot(W1, a0) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = softmax(z2)

        # Backward Pass
        dz2 = a2 - y_train[i].reshape(10, 1)
        dW2 = np.dot(dz2, a1.T)
        db2 = dz2

        dz1 = np.dot(W2.T, dz2) * sigmoid_deriv(z1)
        dW1 = np.dot(dz1, a0.T)
        db1 = dz1

        # Update
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    print(f"Epoch {epoch+1}/{epochs} complete")

# 5. Save the Brain
np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Success! Weights saved to weights.npz")