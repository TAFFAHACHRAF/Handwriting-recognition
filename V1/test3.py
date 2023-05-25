import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

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

def init_params():
    W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(10, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./20)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return W1, b1, W2, b2

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

def open_image():
    file_path = filedialog.askopenfilename(filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")))
    if file_path:
        image = Image.open(file_path).convert("L").resize((28, 28))
        image = np.array(image) / 255.
        W1, b1, W2, b2 = init_params()
        test_prediction(image, W1, b1, W2, b2)

def test_prediction(image, W1, b1, W2, b2):
    current_image = image.flatten()[:, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)

    # Create a new window to display the results
    window = tk.Toplevel(root)
    window.title("Prediction Result")

    prediction_label = tk.Label(window, text="Prediction: " + str(prediction))
    prediction_label.pack()

    image_label = tk.Label(window, image=ImageTk.PhotoImage(image))
    image_label.pack()

def on_submit():
    try:
        index = int(entry.get())
        if index >= 0 and index < m_train:
            W1, b1, W2, b2 = init_params()
            test_prediction(X_train[:, index, None], W1, b1, W2, b2)
        else:
            messagebox.showerror("Error", "Invalid index!")
    except ValueError:
        messagebox.showerror("Error", "Invalid input!")

# Create the GUI
root = tk.Tk()
root.title("Digit Recognition")
root.geometry("300x250")

# Styling
# root.configure(bg="#f0f0ff0")

# Header Label
header_label = tk.Label(root, text="Digit Recognition", font=("Helvetica", 16), bg="#f0f0f0")
header_label.pack(pady=10)

# Image Entry
entry_frame = tk.Frame(root, bg="#f0f0f0")
entry_frame.pack()

entry_label = tk.Label(entry_frame, text="Enter the index of the image to test:", bg="#f0f0f0")
entry_label.pack(side="left")

entry = tk.Entry(entry_frame, width=5, font=("Helvetica", 12))
entry.pack(side="left", padx=5)

# Submit Button
submit_button = tk.Button(root, text="Submit", font=("Helvetica", 12), command=on_submit)
submit_button.pack(pady=10)

# Open Image Button
open_image_button = tk.Button(root, text="Open Image", font=("Helvetica", 12), command=open_image)
open_image_button.pack(pady=10)

# Result Frame
result_frame = tk.Frame(root, bg="#f0f0f0")
result_frame.pack()

# Footer Label
footer_label = tk.Label(root, text="©2023 Digit Recognition. All rights reserved. TAFFAH/ENNANI/BIUCHOUKI/BENHAMAMA",
                        font=("Helvetica", 8), bg="#f0f0f0")
footer_label.pack(pady=5)

root.mainloop()
