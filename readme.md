# Natural Language Processing with Hidden Markov Models

This repository contains Python code for training and using a Hidden Markov Model (HMM) for tagging sequences in text data, particularly focusing on Natural Language Processing (NLP) tasks such as part-of-speech tagging or named entity recognition. The implementation showcases the construction of a vocabulary, model training with transition and emission probabilities, and decoding sequences using a greedy algorithm.

## Features

- **Data Preprocessing**: Functions to load and preprocess text data, including building a vocabulary from training data with a specified frequency threshold to manage rare words.
- **Model Training**: An HMM is trained using input text data, calculating transition and emission probabilities essential for sequence tagging.
- **Decoding**: Implementation of a Greedy Decoder to predict labels for new text sequences based on the trained HMM.
- **Accuracy Evaluation**: Functionality to calculate the accuracy of the model predictions against a test dataset.

## Requirements

- Python 3.x
- Pandas
- NumPy

## Setup and Execution

1. **Clone the Repository**: Start by cloning this repository to your local machine.
   
2. **Install Dependencies**: Ensure that Python 3.x is installed and then install the required packages:
   
   ```bash
   pip install pandas numpy
