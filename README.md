# Neural Network Implementation from Scratch

## Objective:
Implement a simple feedforward neural network from scratch in Python without using any in-built deep learning libraries. This implementation will focus on basic components like forward pass, backward propagation (backpropagation), and training using gradient descent.

---

## Problem Definition

### Dataset:
- **Input (X)**: `[[0, 0], [0, 1], [1, 0], [1, 1]]`
- **Output (y)**: `[[0], [0], [0], [1]]`

This dataset represents the binary inputs and their corresponding output for the AND operation.

### Task:
The task is to train a neural network to predict the output of the AND operation based on the provided input combinations. The problem is a binary classification task.

---

## Methodology

### Neural Network Architecture:
- **Input Layer**: 2 neurons (representing two binary inputs)
- **Hidden Layer**: 3 neurons with sigmoid activation
- **Output Layer**: 1 neuron with sigmoid activation for binary classification

### Forward Pass:
1. The input is passed to the hidden layer, where weighted sums are computed.
2. The hidden layer's output is computed using the sigmoid activation function.
3. The output layer takes the hidden layer's output and computes the final output using sigmoid activation.

### Backpropagation:
- The error is calculated between the predicted output and actual output.
- The error is propagated backward through the network, adjusting the weights and biases using gradient descent based on the computed gradients.

### Loss Function:
- The **Mean Squared Error (MSE)** loss function is used to measure the difference between the predicted and actual outputs.

### Optimization:
- The network is trained using **gradient descent**, adjusting weights and biases in the direction of the negative gradient to minimize the loss function.

---

## How to Run the Code

1. **Clone the repository**:
    ```bash
    git clone https://github.com/sahilkarne/Deep-Learning/tree/main/Neural_Network_From_Scratch.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd Neural_Network_From_Scratch
    ```

3. **Run the Python script**:
    ```bash
    python neural_network.py
    ```

   The network will train on the AND problem dataset for 10,000 epochs and print the loss at regular intervals (every 1000 epochs).

---

## Results

After training, the model will output the predictions for the AND operation based on the input data, showing the predicted results for each input combination.

---

## Technologies Used:
- **Python 3.x**
- **NumPy** (for matrix operations)

---

## Author

- **Sahil Karne**

---

## Acknowledgments
- This implementation is inspired by fundamental concepts in neural networks and machine learning.
