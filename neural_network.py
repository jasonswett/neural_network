import numpy as np

# The sigmoid function maps any real-valued number into the (0, 1) interval,
# making it suitable for predicting probabilities. It's crucial for tasks like
# logistic regression in binary classification problems.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# The derivative of the sigmoid function is used in backpropagation. It helps
# compute the updates to weights necessary for learning by determining the
# gradient of the loss function.
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize the number of neurons in each layer. The input layer has two
# neurons for the XOR inputs. The hidden layer also has two neurons, adjustable
# for complexity. The output layer has one neuron for binary output (0 or 1).
input_layer_neuron_count = 2
hidden_layer_neuron_count = 2
output_layer_neuron_count = 1

# Randomly initialize weights and biases with a uniform distribution. Weights
# adjust during training and determine how neurons influence each other. Biases
# help the model fit better by adjusting the activation function.
np.random.seed(0)
weights_input_to_hidden = np.random.uniform(size=(input_layer_neuron_count,
                                                  hidden_layer_neuron_count))
weights_hidden_to_output = np.random.uniform(size=(hidden_layer_neuron_count,
                                                   output_layer_neuron_count))
bias_hidden_layer = np.random.uniform(size=(1, hidden_layer_neuron_count))
bias_output_layer = np.random.uniform(size=(1, output_layer_neuron_count))

# XOR inputs and expected outputs. XOR only outputs true when inputs differ,
# demonstrating a function that isn't linearly separable without a hidden layer.
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([[0], [1], [1], [0]])

# Training parameters: epochs are the total number of passes through the dataset,
# and learning rate controls the adjustment size to the model during each update.
epochs = 10000
learning_rate = 0.1

# Training loop, involving both forward and backward propagation to make
# predictions and update the model based on errors, respectively.
for epoch in range(epochs):
    # Forward propagation: calculate inputs and activations for hidden layer.
    hidden_layer_input = np.dot(input_data, weights_input_to_hidden) + bias_hidden_layer
    hidden_layer_output = sigmoid(hidden_layer_input)

    # Calculate inputs and activations for the output layer.
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output_layer
    predicted_output = sigmoid(output_layer_input)

    # Calculate loss using Mean Squared Error (MSE), which quantifies the difference
    # between the predicted and actual outputs.
    loss = np.mean((target_output - predicted_output) ** 2)

    # Backpropagation: compute output error and gradient.
    error = target_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    # Calculate hidden layer error and gradient.
    error_hidden_layer = d_predicted_output.dot(weights_hidden_to_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases using the gradients multiplied by the learning rate.
    weights_hidden_to_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output_layer += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_to_hidden += input_data.T.dot(d_hidden_layer) * learning_rate
    bias_hidden_layer += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Periodically output the loss to monitor training progress.
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} Loss {loss}")

# Print the final learned weights and biases.
print("Final weights and biases:")
print("Weights input to hidden:", weights_input_to_hidden)
print("Bias hidden layer:", bias_hidden_layer)
print("Weights hidden to output:", weights_hidden_to_output)
print("Bias output layer:", bias_output_layer)

# Define a function to make predictions with the trained model.
def predict(input_data, weights_input_to_hidden, bias_hidden_layer,
            weights_hidden_to_output, bias_output_layer):
    # Perform forward propagation to calculate the network's predictions.
    hidden_layer_input = np.dot(input_data, weights_input_to_hidden) + bias_hidden_layer
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output_layer
    predicted_output = sigmoid(output_layer_input)
    return predicted_output

# Test the network with the standard XOR inputs.
print("\nTesting the network on new inputs:")
for test_input in np.array([[0, 0], [0, 1], [1, 0], [1, 1]]):
    output = predict(test_input, weights_input_to_hidden, bias_hidden_layer,
                     weights_hidden_to_output, bias_output_layer)
    print(f"Input: {test_input} Output: {output}")
