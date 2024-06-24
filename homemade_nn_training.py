
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os

# Setting up initial variables
learn_rate = 0.01
epochs = 100

# Creates a variable that contains the .txt file that stores the weights and biases
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
variable_file = 'variables.txt'
file_path = os.path.join(CURRENT_DIRECTORY, variable_file)

# Define the size of the neural network
input_size = 784
hidden_size = 20
output_size = 10


def get_mnist(): # Load the MNIST data that will be used for training
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

def load_variables(): # Loads the weights and biases from a file if it exists, otherwise randomize them
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = file.read()
        sections = data.split('Array ')
        arrays = []
        for section in sections[1:]:
            lines = section.strip().split('\n')[1:]
            array = np.loadtxt(lines)
            arrays.append(array)

        weights1, weights2, bias1, bias2 = arrays

        weights1 = weights1.reshape((hidden_size, input_size))
        weights2 = weights2.reshape((output_size, hidden_size))
        bias1 = bias1.reshape((hidden_size, 1))
        bias2 = bias2.reshape((output_size,1))

        print('Loaded Variables')

    else:
        weights1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        weights2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
        bias1 = np.zeros((hidden_size, 1))
        bias2 = np.zeros((output_size, 1))

        print('Randomized Variables')

    return weights1, weights2, bias1, bias2

def save_variables(weights1, weights2, bias1, bias2): # Saves the current values of the weights and biases to be used later

    with open(file_path, 'w') as file:
        pass

    with open(file_path, 'w') as file:
        np.savetxt(file, weights1, header='Array WIH', comments='')
        file.write('\n')
        np.savetxt(file, weights2, header='Array WHO', comments='')
        file.write('\n')
        np.savetxt(file, bias1, header='Array BIH', comments='')
        file.write('\n')
        np.savetxt(file, bias2, header='Array BHO', comments='')
        file.write('\n')  

def relu(x): # Defines the ReLU activation function
    return np.maximum(0, x)

def relu_derivative(x): # Defines derivative of the ReLU function
    return np.where(x > 0, 1, 0)

def sigmoid(x): # Defines the Sigmoid activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): # Defines the derivative of the Sigmoid function
    sig = sigmoid(x)
    return sig * (1 - sig)

def forward_propagation(X): # Run the inputs through the neural network
    Z1 = np.dot(weights1, X) + bias1
    A1 = relu(Z1)
    Z2 = np.dot(weights2, A1) + bias2
    A2 = sigmoid(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

def compute_loss(Y, A2): # Compute the loss using binary cross-entropy loss
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return loss

def compute_accuracy(Y, A2): # Compute the accuracy
    predictions = (A2 > 0.5).astype(int)
    accuracy = np.mean(predictions == Y)
    return accuracy

def backward_propagation(X, Y, cache): # The backward pass that returns the gradients to update variables
    m = X.shape[1]
    Z1, A1, Z2, A2 = cache
    
    dZ2 = A2 - Y
    dweights2 = np.dot(dZ2, A1.T) / m
    dbias2 = np.sum(dZ2, axis=1, keepdims=True) / m
    
    dA1 = np.dot(weights2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dweights1 = np.dot(dZ1, X.T) / m
    dbias1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    gradients = {"dweights1": dweights1, "dbias1": dbias1, "dweights2": dweights2, "dbias2": dbias2}
    return gradients

def update_parameters(params, grads, learning_rate): # Update the weights and biases
    params["weights1"] -= learning_rate * grads["dweights1"]
    params["bias1"] -= learning_rate * grads["dbias1"]
    params["weights2"] -= learning_rate * grads["dweights2"]
    params["bias2"] -= learning_rate * grads["dbias2"]
    return params

def train(x_train, y_train, x_test, y_test, epochs, learning_rate): # The main training loop
    global weights1, bias1, weights2, bias2

    weights1, weights2, bias1, bias2 = load_variables()
    params = {"weights1": weights1, "bias1": bias1, "weights2": weights2, "bias2": bias2}
    
    for i in range(epochs):
        A2, cache = forward_propagation(x_train)
        loss = compute_loss(y_train, A2)
        grads = backward_propagation(x_train, y_train, cache)
        params = update_parameters(params, grads, learning_rate)
        
        weights1, bias1, weights2, bias2 = params["weights1"], params["bias1"], params["weights2"], params["bias2"]
        
        if i % 10 == 0:
            train_accuracy = compute_accuracy(y_train, A2)
            test_A2, _ = forward_propagation(x_test)
            test_accuracy = compute_accuracy(y_test, test_A2)
            print(f"Iteration {i}, Loss: {loss}") 
            print(f"Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    
    
    return params


if __name__ == '__main__': # The start function that loads all the data and variables

    # Load the training data
    images, labels = get_mnist()

    # Shuffle the data so that the traning/test split is different each time
    combined = list(zip(images, labels))
    np.random.shuffle(combined)
    images, labels = zip(*combined)
    images = np.array(images)
    labels = np.array(labels)

    # Split the data into training and test sets
    x_train = images[:48000,].T
    x_test = images[48000:,].T
    y_train = labels[:48000,].T
    y_test = labels[48000:,].T

    trained_params = train(x_train, y_train, x_test, y_test, epochs, learn_rate)

    save_variables(weights1, weights2, bias1, bias2)

    
