
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os


learn_rate = 0.01
number_correct = 0
epochs = 100
batches = 20

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
variable_file = 'variables.txt'
file_path = os.path.join(CURRENT_DIRECTORY, variable_file)
input_size = 784
hidden_size = 20
output_size = 10

W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))


def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

def load_variables():
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        sections = data.split('Array ')
        arrays = []
        for section in sections[1:]:
            lines = section.strip().split('\n')[1:]
            array = np.loadtxt(lines)
            arrays.append(array)

        W1, W2, b1, b2 = arrays

        W1 = W1.reshape((20, 784))
        W2 = W2.reshape((10, 20))
        b1 = b1.reshape((20, 1))
        b2 = b2.reshape((10,1))

        print('Loaded Variables')

    else:
        W1 = np.random.uniform(-0.5, 0.5, (20, 784))
        W2 = np.random.uniform(-0.5, 0.5, (10, 20))
        b1 = np.zeros((20, 1))
        b2 = np.zeros((10, 1))

        print('Randomized Variables')


    return W1, W2, b1, b2

def save_variables(W1, W2, b1, b2):

    with open(file_path, 'w') as file:
        pass

    with open(file_path, 'w') as file:
        np.savetxt(file, W1, header='Array WIH', comments='')
        file.write('\n')
        np.savetxt(file, W2, header='Array WHO', comments='')
        file.write('\n')
        np.savetxt(file, b1, header='Array BIH', comments='')
        file.write('\n')
        np.savetxt(file, b2, header='Array BHO', comments='')
        file.write('\n')  

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def forward_propagation(X):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

# Compute the loss (binary cross-entropy loss)
def compute_loss(Y, A2):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return loss

# Compute the accuracy
def compute_accuracy(Y, A2):
    predictions = (A2 > 0.5).astype(int)
    accuracy = np.mean(predictions == Y)
    return accuracy

# Backward pass
def backward_propagation(X, Y, cache):
    m = X.shape[1]
    Z1, A1, Z2, A2 = cache
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

# Update parameters
def update_parameters(params, grads, learning_rate):
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params

def train(x_train, y_train, x_test, y_test, num_iterations, learning_rate):
    global W1, b1, W2, b2

    W1, W2, b1, b2 = load_variables()
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    for i in range(num_iterations):
        A2, cache = forward_propagation(x_train)
        loss = compute_loss(y_train, A2)
        grads = backward_propagation(x_train, y_train, cache)
        params = update_parameters(params, grads, learning_rate)
        
        W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
        
        if i % 10 == 0:
            train_accuracy = compute_accuracy(y_train, A2)
            test_A2, _ = forward_propagation(x_test)
            test_accuracy = compute_accuracy(y_test, test_A2)
            print(f"Iteration {i}, Loss: {loss}") 
            print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    
    
    return params


if __name__ == '__main__':

    images, labels = get_mnist()
    combined = list(zip(images, labels))
    np.random.shuffle(combined)
    images, labels = zip(*combined)
    images = np.array(images)
    labels = np.array(labels)

    x_train = images[:48000,].T
    x_test = images[48000:,].T
    y_train = labels[:48000,].T
    y_test = labels[48000:,].T

    trained_params = train(x_train, y_train, x_test, y_test, num_iterations=100, learning_rate=0.01)

    save_variables(W1, W2, b1, b2)

    
