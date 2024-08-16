import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize parameters
input_size, hidden_size, output_size = 2, 2, 1
np.random.seed(0)
weights1 = np.random.uniform(size=(input_size, hidden_size))
weights2 = np.random.uniform(size=(hidden_size, output_size))
bias1 = np.random.uniform(size=(1, hidden_size))
bias2 = np.random.uniform(size=(1, output_size))

# Training data for XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

# Training loop
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights1) + bias1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights2) + bias2
    predicted_output = sigmoid(output_layer_input)
    
    # Calculate loss (Mean Squared Error)
    loss = np.mean((Y - predicted_output) ** 2)
    
    # Backpropagation
    error = Y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(weights2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    weights2 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights1 += X.T.dot(d_hidden_layer) * learning_rate
    bias1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} Loss {loss}")

print()
print("Final weights and biases:")
print("Weights1:", weights1)
print("Bias1:", bias1)
print("Weights2:", weights2)
print("Bias2:", bias2)

def predict(X, weights1, bias1, weights2, bias2):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights1) + bias1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights2) + bias2
    predicted_output = sigmoid(output_layer_input)
    return predicted_output

test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print()

# Predict and print the output for test inputs
for test_input in test_inputs:
    output = predict(test_input, weights1, bias1, weights2, bias2)
    print(f"Input: {test_input} Output: {output}")
