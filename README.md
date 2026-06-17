# Hamming Network for Pattern Recognition

## Overview

This project implements a Hamming Network for recognizing binary patterns, with a focus on classifying uppercase English letters represented as 8x8 binary images.

The project includes a basic Hamming Network implementation and an improved version designed to handle noisy or slightly shifted input patterns more robustly. The network compares input patterns to stored training patterns and returns the closest matching class using similarity-based classification.

This project was developed as part of an academic neural networks assignment and demonstrates pattern recognition, binary vector representation, similarity scoring, and robustness testing.

## Main Features

* Pattern recognition using a Hamming Network
* Classification of uppercase letters represented as binary vectors
* One-hot encoded output labels
* Testing on clean input patterns
* Testing with noisy input patterns
* Testing with shifted letter patterns
* Improved similarity calculation for more tolerant pattern matching
* Colored terminal output for easier result analysis

## Project Files

```text
.
├── hamming_net.py
├── improve_hamming.py
├── letters_train.csv
├── letters_test.csv
├── letters_train_shifted_right_padded.csv
├── Hamming network.pptx
└── README.md
```

## Implementation Details

### Basic Hamming Network

The basic implementation stores binary training patterns and compares a given input pattern to all stored patterns.

For each input, the network calculates a similarity score against every stored pattern and returns the label of the closest match.

### Improved Hamming Network

The improved implementation represents each input pattern as an 8x8 matrix and uses a more flexible similarity calculation.

In addition to checking exact pixel matches, the improved version also considers nearby pixels. This helps the network handle small shifts or distortions in the input pattern.

## Dataset

The project uses CSV files containing binary representations of letters:

* `letters_train.csv` - training patterns
* `letters_test.csv` - test patterns
* `letters_train_shifted_right_padded.csv` - shifted patterns used to test robustness

Each letter is represented as a binary pattern, and the output label is represented using one-hot encoding.

## How It Works

1. Load binary letter patterns from CSV files.
2. Store the training patterns in the Hamming Network.
3. Compare each test pattern to the stored patterns.
4. Compute similarity scores.
5. Select the closest matching pattern.
6. Convert the predicted one-hot vector into a letter.
7. Evaluate the prediction against the expected label.

## Technologies Used

* Python
* NumPy
* Pandas
* Pattern Recognition
* Neural Networks
* Similarity-Based Classification

## How to Run

Install the required packages:

```bash
pip install numpy pandas colorama
```

Run the basic implementation:

```bash
python hamming_net.py
```

Run the improved implementation:

```bash
python improve_hamming.py
```

## What I Learned

This project helped me understand how simple neural network models can be used for pattern recognition tasks.

Through this project, I practiced:

* Representing visual patterns as binary vectors
* Implementing a Hamming Network from scratch
* Working with one-hot encoded labels
* Comparing patterns using similarity scores
* Testing model robustness with noisy and shifted inputs
* Improving a baseline implementation through more flexible matching logic

## Future Improvements

Possible future improvements include:

* Adding automated accuracy comparison between the basic and improved implementations
* Visualizing the 8x8 letter patterns
* Adding more distorted test examples
* Supporting additional pattern categories
* Refactoring the code into reusable modules
* Adding unit tests for the similarity and classification functions
