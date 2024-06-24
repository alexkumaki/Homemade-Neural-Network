# Homemade-Neural-Network

Neural Networks seem really complicated. This is probably because of the heavy use of buzzwords and complex metaphors about teaching computers to think like humans. However, Neural Networks are just an optimization problem which is to say: math.

The idea behind this project is to read in a user drawing of a single digit, and let the user know what digit they were probably trying to draw. It reads it in as a 28x28 pixel image (so 784 inputs), pushes it through a 20-node hidden layer, and outputs one of 10 possible outputs. The network is trained on the MNIST data file, which provides 60,000 28x28 images of single digits and their associated labels. It saves the weights and bias variables in a .txt file so that you don't have to re-train it every time.

Note: This network is bad so far. It probably needs a larger hidden layer and needs to be trained on a more noisy and varied training set, both of which will be implemented at some point. 

TODO:
  - Add comments to code
  - Increase size of hidden layer?
  - Add random noise and adjustments to increase accuracy
