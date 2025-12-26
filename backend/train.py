import numpy as np
import tensorflow as tf

# Load and Preprocess Data
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()
X_train = X_train_raw.reshape(-1, 784) / 255.0
y_train = np.eye(10)[y_train_raw]

# Architecture: 784 -> 128 -> 10
input_size = 784
hidden_size = 128  # Increased Density
output_size = 10

# He Initialization (Better for ReLU)
W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
b2 = np.zeros((output_size, 1))

def relu(z): return np.maximum(0, z)
def relu_deriv(z): return z > 0
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum(axis=0)

# Training Hyperparameters
learning_rate = 0.01
lambda_reg = 0.001  # Weight Decay (L2 Regularization)
epochs = 15

print(f"Training with {hidden_size} neurons and ReLU...")
for epoch in range(epochs):
    permutation = np.random.permutation(len(X_train))
    X_train, y_train = X_train[permutation], y_train[permutation]

    for i in range(len(X_train)):
        # Forward Pass
        a0 = X_train[i].reshape(784, 1)
        z1 = np.dot(W1, a0) + b1
        a1 = relu(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = softmax(z2)

        # Backpropagation
        dz2 = a2 - y_train[i].reshape(10, 1)
        dW2 = np.dot(dz2, a1.T) + (lambda_reg * W2) # Added Regularization
        db2 = dz2

        dz1 = np.dot(W2.T, dz2) * relu_deriv(z1)
        dW1 = np.dot(dz1, a0.T) + (lambda_reg * W1) # Added Regularization
        db1 = dz1

        # Parameter Updates
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    print(f"Epoch {epoch + 1}/{epochs} complete")

np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("New weights saved!")