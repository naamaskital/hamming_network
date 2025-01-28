
class HammingNetwork:
    def __init__(self, input_vectors, output_vectors):
        """
        Initialize the Hamming Network.
        :param input_vectors: List of input patterns (64 pixels).
        :param output_vectors: List of output patterns (OneHot encoded).
        """
        self.input_vectors = np.array(input_vectors).reshape(-1, 8, 8)  # Reshape inputs to 8x8 matrices
        self.output_vectors = np.array(output_vectors)  # Store output patterns

    def compute_similarity(self, input_matrix):
        """
        Compute similarity between the input and all stored patterns pixel-by-pixel.
        Add 1 if pixels match (both 0 and 1); if not, check adjacent pixels.
        :param input_matrix: Input matrix to compare (8x8).
        :return: List of similarity scores for each stored pattern.
        """
        similarities = []

        for pattern in self.input_vectors:  # Iterate over stored patterns
            score = 0
            for i in range(8):  # Loop through rows
                for j in range(8):  # Loop through columns
                    if input_matrix[i][j] == pattern[i][j]:
                        score += 1  # Add 1 if pixels match
                    else:
                        # Check adjacent pixels: left, right, up, down, and diagonals
                        for di, dj in [
                            (-1, 0), (1, 0),  # Up and down
                            (0, -1), (0, 1),  # Left and right
                            (-1, -1), (-1, 1),  # Upper diagonals
                            (1, -1), (1, 1)  # Lower diagonals
                        ]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 8 and 0 <= nj < 8:  # Check boundaries
                                if input_matrix[ni][nj] == pattern[i][j]:
                                    score += 0.7  # Add 0.7 for matching neighbors
                                    break  # Stop checking neighbors for this pixel
            similarities.append(score)

        return similarities

    def classify(self, input_vectors):
        """
        Identify the closest matching pattern in memory.
        :param input_matrix: Input matrix to classify (8x8).
        :return: Closest matching output pattern.
        """
        similarities = self.compute_similarity(input_vectors)
        best_match_index = np.argmax(similarities)  # Find the pattern with the highest similarity
        return self.output_vectors[best_match_index]


import random
import numpy as np
import pandas as pd
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init(autoreset=True)

# Load the training dataset
data = pd.read_csv('letters_train.csv')
inputs = data.iloc[:, :64].values.astype(int).reshape(-1, 8, 8)  # Reshape to 8x8 matrices
outputs = data.iloc[:, 65:].values.astype(int)  # Extract output patterns

# Function to add noise to the inputs
def add_noise(matrix, noise_level):
    """
    Add random noise to the input matrix by flipping bits.
    :param matrix: Input matrix to modify (8x8).
    :param noise_level: Fraction of bits to flip.
    :return: Noisy version of the input matrix.
    """
    noisy_matrix = matrix.copy()
    n_noisy_bits = int(noise_level * matrix.size)  # Calculate number of bits to flip
    indices = random.sample(range(matrix.size), n_noisy_bits)  # Randomly select bits to flip
    for index in indices:
        row, col = divmod(index, matrix.shape[1])  # Map linear index to 2D indices
        noisy_matrix[row, col] = 1 - noisy_matrix[row, col]  # Flip bit (0 -> 1 or 1 -> 0)
    return noisy_matrix

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
    :param test_inputs: Test input matrices.
    :param test_outputs: Expected output patterns.
    :param noise_level: Fraction of noise to add to inputs (optional).
    :return: Accuracy of the network on the test data.
    """
    if noise_level is not None:
        print(f"\nTesting with {int(noise_level * 100)}% noise:")

    correct_count = 0
    for i, input_matrix in enumerate(test_inputs):
        # Add noise if required
        if noise_level is not None:
            input_matrix = add_noise(input_matrix, noise_level)

        predicted_output = hamming_net.classify(input_matrix)
        expected_output = test_outputs[i]

        predicted_char = onehot_to_char(predicted_output)  # Convert predicted output to character
        expected_char = onehot_to_char(expected_output)  # Convert expected output to character

        is_correct = np.array_equal(predicted_output, expected_output)  # Check if prediction matches
        color = Fore.GREEN if is_correct else Fore.RED  # Use green for correct and red for incorrect

        print(
            f"{color}Example {i + 1}: {'Correct' if is_correct else 'Incorrect'} "
            f"(Predicted: {predicted_char}, Expected: {expected_char}){Style.RESET_ALL}"
        )

        if is_correct:
            correct_count += 1

    accuracy = correct_count / len(test_inputs)  # Calculate accuracy
    print(f"Overall accuracy: {Fore.BLUE}{accuracy:.2%}{Style.RESET_ALL}")
    return accuracy

# Initialize the Hamming Network
hamming_net = HammingNetwork(inputs, outputs)

# Test on noisy training data
noise_levels = [0.05, 0.1, 0.2]  # Define noise levels for testing
for noise_level in noise_levels:
    test_on_test_data_with_colors(hamming_net, inputs, outputs, noise_level=noise_level)

print("\nTesting with test data:")

# Load the test dataset
test_data = pd.read_csv('letters_test.csv')
test_inputs = test_data.iloc[:, :64].values.astype(int).reshape(-1, 8, 8)  # Reshape to 8x8 matrices
test_outputs = np.array([list(map(int, onehot.strip('[]').split(','))) for onehot in test_data['OneHot']])  # Convert OneHot to lists

# Test on clean test data
test_on_test_data_with_colors(hamming_net, test_inputs, test_outputs)

print("\nTesting with shifted train data:")

# Test on shifted training data
shifted_data = pd.read_csv('letters_train_shifted_right_padded.csv')
shifted_inputs = shifted_data.iloc[:, :64].values.astype(int).reshape(-1, 8, 8)  # Reshape to 8x8 matrices
shifted_outputs = shifted_data.iloc[:, 64:].values.astype(int)  # Extract output patterns

test_on_test_data_with_colors(hamming_net, shifted_inputs, shifted_outputs)