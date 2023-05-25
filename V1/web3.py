import cv2
import numpy as np
import streamlit as st

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def get_predictions(A3):
    return np.argmax(A3, 0)


def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions


def load_model():
    W1 = np.load('model/tree_layers/W1.npy')
    b1 = np.load('model/tree_layers/b1.npy')
    W2 = np.load('model/tree_layers/W2.npy')
    b2 = np.load('model/tree_layers/b2.npy')
    W3 = np.load('model/tree_layers/W3.npy')
    b3 = np.load('model/tree_layers/b3.npy')
    return W1, b1, W2, b2, W3, b3


def load_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(image, (28, 28))
    flattened_image = resized_image.flatten() / 255.0   # type: ignore
    return flattened_image


uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if uploaded_file is not None:

    W1, b1, W2, b2, W3, b3 = load_model()
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, width=400)
    image = load_image(opencv_image)
    input_image = image.reshape((784, 1))

    if st.button('Predict'):
        prediction = make_predictions(input_image, W1, b1, W2, b2, W3, b3)
        st.write('prediction: ', prediction)
