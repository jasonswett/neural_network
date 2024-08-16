import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

input_layer_neuron_count = 2
hidden_layer_neuron_count = 2
output_layer_neuron_count = 1

np.random.seed(0)
weights_input_to_hidden = np.random.uniform(size=(input_layer_neuron_count, hidden_layer_neuron_count))
weights_hidden_to_output = np.random.uniform(size=(hidden_layer_neuron_count, output_layer_neuron_count))
bias_hidden_layer = np.random.uniform(size=(1, hidden_layer_neuron_count))
bias_output_layer = np.random.uniform(size=(1, output_layer_neuron_count))

# Training data for XOR
input_data = np.array([[0,0], [0,1], [1,0], [1,1]])
target_output = np.array([[0], [1], [1], [0]])

# Training loop
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(input_data, weights_input_to_hidden) + bias_hidden_layer
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output_layer
    predicted_output = sigmoid(output_layer_input)
    
    # Calculate loss (Mean Squared Error)
    loss = np.mean((target_output - predicted_output) ** 2)
    
    # Backpropagation
    error = target_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(weights_hidden_to_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    weights_hidden_to_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output_layer += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_to_hidden += input_data.T.dot(d_hidden_layer) * learning_rate
    bias_hidden_layer += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} Loss {loss}")

print()
print("Final weights and biases:")
print("Weights input to hidden:", weights_input_to_hidden)
print("Bias hidden layer:", bias_hidden_layer)
print("Weights hidden to output:", weights_hidden_to_output)
print("Bias output layer:", bias_output_layer)

def predict(input_data, weights_input_to_hidden, bias_hidden_layer, weights_hidden_to_output, bias_output_layer):
    # Forward propagation
    hidden_layer_input = np.dot(input_data, weights_input_to_hidden) + bias_hidden_layer
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + bias_output_layer
    predicted_output = sigmoid(output_layer_input)
    return predicted_output

test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print()

# Predict and print the output for test inputs
for test_input in test_inputs:
    output = predict(test_input, weights_input_to_hidden, bias_hidden_layer, weights_hidden_to_output, bias_output_layer)
    print(f"Input: {test_input} Output: {output}")
