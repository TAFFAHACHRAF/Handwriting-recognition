import numpy as np
import pandas as pd

def load_data(filename):
    data = pd.read_csv(filename)
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)
    return data, m, n

def split_data(data, m, n, dev_size=1000):
    data_dev = data[:dev_size].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[dev_size:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.

    _, m_train = X_train.shape

    return X_dev, Y_dev, X_train, Y_train, m_train
