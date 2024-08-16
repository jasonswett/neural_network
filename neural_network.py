import numpy as np

# The sigmoid function is a type of activation function that is historically popular in 
# training neural networks. The mathematical form is sigmoid(x) = 1 / (1 + exp(-x)).
# It is particularly useful because it maps any real-valued number into the (0, 1) interval,
# making it suitable for models where we need to predict probabilities. The sigmoid function
# has seen extensive use in binary classification problems, such as logistic regression.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# The derivative of the sigmoid function is used in the backpropagation process, which
# is crucial for training neural networks. The gradient of the sigmoid function helps
# in calculating the updates to weights in the network, essential for learning. The derivative
# sigma'(x) = sigma(x) * (1 - sigma(x)) tells us how to change the weights where the output
# is not close to the actual target value, thus minimizing the loss during training.
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize the number of neurons in each layer of our neural network. The input layer
# will have two neurons, corresponding to the two inputs in the XOR function. The hidden layer
# will also contain two neurons; this number can be adjusted to increase model complexity.
# The output layer contains a single neuron as XOR outputs a single binary value (0 or 1).
input_layer_neuron_count = 2
hidden_layer_neuron_count = 2
output_layer_neuron_count = 1

# Random initialization of weights and biases. Weights are crucial parameters in neural networks
# that adjust during training and determine the strength of the influence one neuron has over another.
# Biases are additional parameters that allow the model to better fit the data by shifting the activation
# function to the left or right, which may be critical for learning patterns.
np.random.seed(0)  # Seed for reproducibility.
weights_input_to_hidden = np.random.uniform(size=(input_layer_neuron_count, hidden_layer_neuron_count))
weights_hidden_to_output = np.random.uniform(size=(hidden_layer_neuron_count, output_layer_neuron_count))
bias_hidden_layer = np.random.uniform(size=(1, hidden_layer_neuron_count))
bias_output_layer = np.random.uniform(size=(1, output_layer_neuron_count))

# XOR input and corresponding output. The XOR (exclusive OR) function outputs true only when inputs differ.
# It is a fundamental example of a function that is not linearly separable, so a simple linear classifier
# cannot learn this function without a hidden layer to introduce non-linearity to the model.
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([[0], [1], [1], [0]])

# Training parameters: epochs define the number of times the learning algorithm will work through
# the entire training dataset. The learning rate controls how much to change the model in response
# to the estimated error each time the model weights are updated. Choosing the right number of epochs
# and learning rate is crucial for effective learning.
epochs = 10000
learning_rate = 0.1

# Training loop: the most crucial part of neural network implementation. This loop involves both forward
# propagation to make predictions and backward propagation to update weights and biases based on the error.
for epoch in range(epochs):
    # Forward Propagation
    # Compute the input to the hidden layer by multiplying the input data with the weights
    # and adding the bias. This combines the input features in different ways before activation,
    # allowing the network to make complex mappings from inputs to outputs.
    hidden_layer_input = np.dot(input_data, weights_input_to_hidden) + bias_hidden_layer
    hidden_layer_output = sigmoid(hidden_layer_input)  # Apply sigmoid activation function.

    # Compute the input to the output layer by multiplying the activated hidden layer output
    # with the weights leading to the output layer and adding the output layer bias.
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output_layer
    predicted_output = sigmoid(output_layer_input)  # Apply sigmoid to generate final output.

    # Calculate the loss using Mean Squared Error (MSE). This quantifies how much the predicted
    # output differs from the actual output, providing a basis for updating weights and biases.
    loss = np.mean((target_output - predicted_output) ** 2)

    # Backpropagation
    # Compute the error of the output (difference between predicted and actual output).
    error = target_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)  # Gradient of the loss function.

    # Calculate the error of the hidden layer. This step propagates the error back from the output
    # to the hidden layer, allowing updates to the weights between the input and hidden layers.
    error_hidden_layer = d_predicted_output.dot(weights_hidden_to_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)  # Gradient at hidden layer.

    # Update the weights and biases using the calculated gradients and the learning rate.
    # This step adjusts the model parameters to minimize the loss function further.
    weights_hidden_to_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output_layer += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_to_hidden += input_data.T.dot(d_hidden_layer) * learning_rate
    bias_hidden_layer += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Periodically output the loss to monitor the training progress.
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} Loss {loss}")

# Output the final weights and biases to see what the network has learned.
print("Final weights and biases:")
print("Weights input to hidden:", weights_input_to_hidden)
print("Bias hidden layer:", bias_hidden_layer)
print("Weights hidden to output:", weights_hidden_to_output)
print("Bias output layer:", bias_output_layer)

# Function to make predictions with the trained model.
def predict(input_data, weights_input_to_hidden, bias_hidden_layer,
            weights_hidden_to_output, bias_output_layer):
    # Forward propagation to predict outputs.
    hidden_layer_input = np.dot(input_data, weights_input_to_hidden) + bias_hidden_layer
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output_layer
    predicted_output = sigmoid(output_layer_input)
    return predicted_output

# Test the network on all possible XOR inputs.
print("\nTesting the network on new inputs:")
for test_input in np.array([[0, 0], [0, 1], [1, 0], [1, 1]]):
    output = predict(test_input, weights_input_to_hidden, bias_hidden_layer,
                     weights_hidden_to_output, bias_output_layer)
    print(f"Input: {test_input} Output: {output}")
