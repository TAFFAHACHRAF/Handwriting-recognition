import numpy as np
import model_utils

def load_modal_2l():
    W1 = np.load('model/W1.npy')
    b1 = np.load('model/b1.npy')
    W2 = np.load('model/W2.npy')
    b2 = np.load('model/b2.npy')
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = model_utils.forward_prop(W1, b1, W2, b2, X)
    predictions = model_utils.get_predictions(A2)
    return predictions
