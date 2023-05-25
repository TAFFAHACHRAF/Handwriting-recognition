import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

data = np.array(data)
data_test = np.array(data_test)

m, n = data.shape
m_test, n_test = data_test.shape

data_dev = data[0:1000].T
data_test = data_test.T
data_train = data[1000:m].T

Y_dev = data_train[0]
X_dev = data_train[1:n]
X_dev = X_dev / 255

X_test = data_test / 255

Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_, m_train = X_train.shape


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# activation function ReLU


def ReLU(Z):
    return np.maximum(Z, 0)


def ReLU_deriv(Z):
    return Z > 0

# activation function Leaky ReLU


def leaky_relu(Z, alpha=0.1):
    return np.maximum(alpha * Z, Z)


def leaky_relu_deriv(Z, alpha=0.1):
    dZ = np.ones_like(Z)
    dZ[Z < 0] = alpha
    return dZ

# activation function softmax


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = leaky_relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * leaky_relu_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(
            W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


def blind_prediction(index, W1, b1, W2, b2):
    current_image = X_test[:, index, None]
    print("current_image", current_image)
    print("current_image.shape", current_image.shape)
    prediction = make_predictions(X_test[:, index, None], W1, b1, W2, b2)
    print("Prediction: ", prediction)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


def save_modal(W1, b1, W2, b2):
    np.save('model/two_layers/W1.npy', W1)
    np.save('model/two_layers/b1.npy', b1)
    np.save('model/two_layers/W2.npy', W2)
    np.save('model/two_layers/b2.npy', b2)


def load_modal():
    W1 = np.load('model/two_layers/W1.npy')
    b1 = np.load('model/two_layers/b1.npy')
    W2 = np.load('model/two_layers/W2.npy')
    b2 = np.load('model/two_layers/b2.npy')
    return W1, b1, W2, b2


if __name__ == "__main__":
    W1, b1, W2, b2 = load_modal()
    test_prediction(0, W1, b1, W2, b2)
    test_prediction(1, W1, b1, W2, b2)
    test_prediction(2, W1, b1, W2, b2)
    test_prediction(3, W1, b1, W2, b2)

    blind_prediction(0, W1, b1, W2, b2)
    blind_prediction(20, W1, b1, W2, b2)
    blind_prediction(40, W1, b1, W2, b2)
    blind_prediction(60, W1, b1, W2, b2)

    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    print(get_accuracy(dev_predictions, Y_dev))
