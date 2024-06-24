
import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageGrab
import os
import pathlib

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
variable_file = 'variables.txt'
file_path = os.path.join(CURRENT_DIRECTORY, variable_file)
input_size = 784
hidden_size = 20
output_size = 10


# Define the activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

        weights1, weights2, bias1, bias2 = arrays

        weights1 = weights1.reshape((20, 784))
        weights2 = weights2.reshape((10, 20))
        bias1 = bias1.reshape((20, 1))
        bias2 = bias2.reshape((10,1))

        print('Loaded Variables')

    else:
        weights1 = np.random.uniform(-0.5, 0.5, (20, 784))
        weights2 = np.random.uniform(-0.5, 0.5, (10, 20))
        bias1 = np.zeros((20, 1))
        bias2 = np.zeros((10, 1))

        print('Randomized Variables')


    return weights1, weights2, bias1, bias2

# Forward pass
def forward_propagation(X):
    Z1 = np.dot(weights1, X) + bias1
    A1 = relu(Z1)
    Z2 = np.dot(weights2, A1) + bias2
    A2 = sigmoid(Z2)
    return A2

# Create the GUI
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        
        self.canvas = Canvas(root, width=200, height=200, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.canvas.bind("<bias1-Motion>", self.paint)
        
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0)
        
        self.predict_button = tk.Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=1, column=1)
        
        self.result_label = tk.Label(root, text="Draw a digit and click Predict")
        self.result_label.grid(row=1, column=2, columnspan=2)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)

    def clear_canvas(self):
        self.canvas.delete("all")

    def predict_digit(self):
        # Capture the canvas content and preprocess the image
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        
        # Grab the image from the canvas
        image = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28)).convert("L")
        
        # Preprocess the image to match the neural network input
        image_np = np.array(image)
        image_np = 255 - image_np  # Invert the colors
        image_np = image_np / 255.0  # Normalize the pixel values
        image_np = image_np.reshape(1, 784).T  # Flatten and transpose to match input shape
        
        # Predict using the neural network
        prediction = forward_propagation(image_np)
        predicted_digit = np.argmax(prediction, axis=0)
        
        # Update the result label
        self.result_label.config(text=f"Predicted Digit: {predicted_digit[0]}")

# Run the application
if __name__ == "__main__":
    weights1, weights2, bias1, bias2 = load_variables()
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()