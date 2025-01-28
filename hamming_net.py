# Basic Hamming Network

class HammingNetwork:
    def __init__(self, input_vectors, output_vectors):
        """
        Initialize the Hamming Network.
        :param input_vectors: List of input patterns (64 pixels).
        :param output_vectors: List of output patterns (OneHot encoded).
        """
        self.input_vectors = np.array(input_vectors)  # Store input patterns in memory
        self.output_vectors = np.array(output_vectors)  # Store output patterns in memory

    def compute_similarity(self, input_vector):
        """
        Compute the similarity between the input and all stored patterns in memory.
        :param input_vector: Input pattern to compare (64 pixels).
        :return: List of similarity scores for each stored pattern.
        """
        similarities = []
        for pattern in self.input_vectors:
            # Compare input with stored pattern pixel by pixel
            score = np.sum(input_vector == pattern)
            similarities.append(score)
        return similarities

    def classify(self, input_vector):
        """
        Identify the closest matching pattern in memory.
        :param input_vector: Input pattern to classify (64 pixels).
        :return: Closest matching output pattern (OneHot encoded).
        """
        similarities = self.compute_similarity(input_vector)
        best_match_index = np.argmax(similarities)  # Find the pattern with the highest similarity score
        return self.output_vectors[best_match_index]


import random
import numpy as np
import pandas as pd
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)

# Load the training dataset
data = pd.read_csv('letters_train.csv')

# Prepare inputs (64 pixels) and outputs (OneHot encoding for 26 letters)
inputs = data.iloc[:, :64].values.astype(int)
outputs = data.iloc[:, 65:].values.astype(int)

# Function to add noise to the inputs
def add_noise(data, noise_level):
    """
    Add random noise to the input data by flipping bits.
    :param data: Input data to modify.
    :param noise_level: Fraction of bits to flip.
    :return: Noisy version of the input data.
    """
    noisy_data = data.copy()
    n_noisy_bits = int(noise_level * len(data))
    indices = random.sample(range(len(data)), n_noisy_bits)
    for index in indices:
        noisy_data[index] = 1 - noisy_data[index]  # Flip bit (0 -> 1 or 1 -> 0)
    return noisy_data

# Convert OneHot vector to a character
def onehot_to_char(onehot_vector):
    """
    Convert a OneHot encoded vector to a character.
    :param onehot_vector: OneHot encoded vector.
    :return: Corresponding character (A-Z).
    """
    index = np.argmax(onehot_vector)  # Find the index of the '1'
    return chr(ord('A') + index)  # Convert index to a letter

# Test the Hamming Network on test data with colored results
def test_on_test_data_with_colors(hamming_net, test_inputs, test_outputs, noise_level=None):
    """
    Test the Hamming Network and display results with color-coded accuracy.
    :param hamming_net: HammingNetwork object.
    :param test_inputs: Test input patterns.
    :param test_outputs: Expected output patterns.
    :param noise_level: Fraction of noise to add to inputs (optional).
    :return: Accuracy of the network on the test data.
    """
    if noise_level is not None:
        print(f"\nTesting with {int(noise_level * 100)}% noise:")

    correct_count = 0
    for i, input_vector in enumerate(test_inputs):
        # Add noise if required
        if noise_level is not None:
            input_vector = add_noise(input_vector, noise_level)

        predicted_output = hamming_net.classify(input_vector)
        expected_output = test_outputs[i]

        # Convert OneHot vectors to characters
        predicted_char = onehot_to_char(predicted_output)
        expected_char = onehot_to_char(expected_output)

        # Check if the prediction is correct
        is_correct = np.array_equal(predicted_output, expected_output)
        color = Fore.GREEN if is_correct else Fore.RED

        # Display the result for the current input
        print(
            f"{color}Example {i + 1}: {'Correct' if is_correct else 'Incorrect'} "
            f"(Predicted: {predicted_char}, Expected: {expected_char}){Style.RESET_ALL}"
        )

        if is_correct:
            correct_count += 1

    # Display overall accuracy
    accuracy = correct_count / len(test_inputs)
    print(f"Overall accuracy: {Fore.BLUE}{accuracy:.2%}{Style.RESET_ALL}")
    return accuracy

# Initialize the Hamming Network with the training data
hamming_net = HammingNetwork(inputs, outputs)

# Test on noisy test data
noise_levels = [0.05, 0.1, 0.2]
for noise_level in noise_levels:
    test_on_test_data_with_colors(hamming_net, inputs, outputs, noise_level=noise_level)

print("\nTesting with test data:")

# Load the test dataset
test_data = pd.read_csv('letters_test.csv')

# Prepare the test inputs and outputs
test_inputs = test_data.iloc[:, :64].values.astype(int)  # First 64 columns as input (pixels)
test_outputs = np.array([eval(onehot) for onehot in test_data['OneHot']])  # Convert OneHot column to lists

# Test on the clean test data
test_on_test_data_with_colors(hamming_net, test_inputs, test_outputs)
print("\nTesting with shifted train data:")

# Load the shifted training dataset
shifted_data = pd.read_csv('letters_train_shifted_right_padded.csv')
shifted_inputs = shifted_data.iloc[:, :64].values.astype(int)
shifted_outputs = shifted_data.iloc[:, 64:].values.astype(int)

# Test on the shifted training data
test_on_test_data_with_colors(hamming_net, shifted_inputs, shifted_outputs)
