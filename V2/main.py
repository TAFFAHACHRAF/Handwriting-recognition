import cv2
import numpy as np
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

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

def load_modal_2l():
    W1 = np.load('model/W1.npy')
    b1 = np.load('model/b1.npy')
    W2 = np.load('model/W2.npy')
    b2 = np.load('model/b2.npy')
    return W1, b1, W2, b2

def load_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(image, (28, 28))
    flattened_image = resized_image.flatten() / 255.0
    return flattened_image

W1, b1, W2, b2 = load_modal_2l()

st.set_option('deprecation.showPyplotGlobalUse', False)

def test_prediction(index):
    current_image = X_train[:, index-1, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    st.pyplot()  # Display the plot using st.pyplot() instead of plt.show()

test_prediction(0)

uploaded_file = st.file_uploader("Choose an image file", type="jpg")

if uploaded_file is not None:
    W1, b1, W2, b2 = load_modal_2l()
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, width=400)
    image = load_image(opencv_image)
    input_image = image.reshape((784, 1))

    if st.button('Predict'):
        print("Input Image Shape: ", input_image.shape)
        print("Input Image Range: ", np.min(input_image), np.max(input_image))
        prediction = make_predictions(input_image, W1, b1, W2, b2)
        print("Prediction: ", prediction)