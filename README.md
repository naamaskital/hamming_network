# Hamming Network Project

## Overview
This project implements a Hamming neural network designed for recognizing patterns, such as English uppercase letters, represented in binary format. The implementation includes a standard Hamming network and an enhanced version that improves performance and error tolerance. The network is trained and tested using provided datasets.

## Features
- **hamming_net.py**: Contains the core implementation of the standard Hamming network.
- **improve_hamming.py**: Introduces an improved version of the Hamming network, optimizing recall accuracy and error handling.
- **Hamming network.pptx**: A presentation explaining the network's theory, implementation, and performance.
- **letters_train.csv**: Dataset containing training patterns for the network.
- **letters_train_shifted_right_padded.csv**: A modified version of the training data to test the network's adaptability.
- **letters_test.csv**: Dataset used for testing the network's recall capability.

## Structure
```
hamming_network/
├── hamming_net.py
├── improve_hamming.py
├── Hamming network.pptx
├── letters_train.csv
├── letters_train_shifted_right_padded.csv
├── letters_test.csv
├── .git
├── .gitattributes
```

### File Descriptions
- **hamming_net.py**: Implements the standard Hamming network algorithm using binary vectors for pattern recognition.
- **improve_hamming.py**: Enhances the standard implementation with features such as:
  - Increased tolerance for noisy inputs.
  - Optimized weight calculation and pattern matching.
  - Adaptive learning for dynamic data.
- **Hamming network.pptx**: Explains the network's concept and provides visual insights into its implementation.
- **letters_train.csv**: Provides training data for the network, including binary representations of patterns.
- **letters_train_shifted_right_padded.csv**: Modified training data for testing robustness.
- **letters_test.csv**: Test data for evaluating the network's performance.

## Enhancements in `improve_hamming.py`
- **Error Tolerance**: Handles noisy or incomplete patterns more effectively.
- **Optimized Recall**: Improves the accuracy and speed of pattern recognition.
- **Dynamic Learning**: Supports incremental training with new patterns.

## Usage
### Prerequisites
- Python 3.7 or higher.
- Required libraries: NumPy, pandas.

Install the required libraries using pip:
```bash
pip install numpy pandas
```

### Training and Testing
1. **Standard Hamming Network**:
   Run the `hamming_net.py` script:
   ```bash
   python hamming_net.py
   ```
   This script trains the Hamming network using `letters_train.csv` and tests it with `letters_test.csv`.

2. **Enhanced Hamming Network**:
   Run the `improve_hamming.py` script:
   ```bash
   python improve_hamming.py
   ```
   This script trains the enhanced network and demonstrates improved recall performance with noisy or modified test data.

### Input and Output
- **Input**: Binary patterns representing letters or other data.
- **Output**: Predicted patterns or performance metrics.

## Example
### Standard Hamming Network
Input: Noisy representation of a pattern.
Output: Correctly recalled pattern or closest match.

### Enhanced Hamming Network
Input: Heavily distorted or padded representation of a pattern.
Output: Accurate recall due to improved error handling.



