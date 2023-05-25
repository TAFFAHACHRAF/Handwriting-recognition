import cv2
import streamlit as st
import numpy as np
import data_utils
import prediction  # Rename the import

import matplotlib.pyplot as plt

data, m, n = data_utils.load_data('data/train.csv')
X_dev, Y_dev, X_train, Y_train, m_train = data_utils.split_data(data, m, n)
W1, b1, W2, b2 = prediction.load_modal_2l()  # Update the function call

st.set_option('deprecation.showPyplotGlobalUse', False)

def test_prediction(index):
    current_image = X_train[:, index, None]
    result = prediction.make_predictions(X_train[:, index, None], W1, b1, W2, b2)  # Rename the variable
    label = Y_train[index]
    print("Prediction: ", result)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

while True:
    index = int(input("Enter a number (0 - 59999): "))
    test_prediction(index)
