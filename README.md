To run the project:

1. Install the required dependencies.
2. Download the dataset files from the `data` directory.
3. Open a terminal or command prompt.
4. Navigate to the project directory.
5. Run the `main.py` script.

The project implements a neural network with a two-layer architecture. The architecture consists of an input layer, a hidden layer with ReLU activation, and an output layer with softmax activation. The weights and biases of the network are loaded from the `model` directory.

Here's an example of an important function in the code:

python
### `forward_prop(W1, b1, W2, b2, X)`

Performs forward propagation in the neural network.

**Arguments:**

- `W1`: Weight matrix for the first layer.
- `b1`: Bias vector for the first layer.
- `W2`: Weight matrix for the second layer.
- `b2`: Bias vector for the second layer.
- `X`: Input data.

**Returns:**

- `Z1`: Output of the first layer linear transformation.
- `A1`: Output of the first layer after applying the ReLU activation function.
- `Z2`: Output of the second layer linear transformation.
- `A2`: Output of the second layer after applying the softmax activation function.
You can customize and add more function documentation as needed for your project.

Data visualisation
![data_visualization](demo/data.jpeg)


## Project Structure

The project is structured as follows:
```
C:.
├── README.md
└── V1
├── main.py
├── training.py
├── data
│ ├── sample_submission.csv
│ ├── test.csv
│ └── train.csv
└── model
├── W1.npy
├── b1.npy
├── W2.npy
└── b2.npy
```






