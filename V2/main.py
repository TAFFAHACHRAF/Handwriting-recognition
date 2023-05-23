import cv2
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

data = pd.read_csv('data/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape
Y_train


def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def load_model_2l():
    W1 = np.load('model/W1.npy')
    b1 = np.load('model/b1.npy')
    W2 = np.load('model/W2.npy')
    b2 = np.load('model/b2.npy')
    return W1, b1, W2, b2

def load_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(image, (28, 28))
    flattened_image = resized_image.flatten() / 255.0  # Normalize the image
    return flattened_image

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

W1, b1, W2, b2 = load_model_2l()

st.set_option('deprecation.showPyplotGlobalUse', False)

index = st.number_input("Enter an index:", min_value=0, max_value=len(Y_train)-1, step=1)
# if st.button('Predict'):
#     prediction = make_predictions(index, W1, b1, W2, b2)
#     true_label = Y_train[index]
#     predicted_label = prediction[0]
#     st.write("True Label: ", true_label)
#     st.write("Predicted Label: ", predicted_label)

while True:
    index = int(input("Enter a number (0 - 59999): "))
    test_prediction(index, W1, b1, W2, b2)