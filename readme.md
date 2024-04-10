# Natural Language Processing with Hidden Markov Models

This repository contains Python code for training and using a Hidden Markov Model (HMM) for tagging sequences in text data, particularly focusing on Natural Language Processing (NLP) tasks such as part-of-speech tagging or named entity recognition. The implementation showcases the construction of a vocabulary, model training with transition and emission probabilities, and decoding sequences using a greedy algorithm.

## Features

- **Data Preprocessing**: Functions to load and preprocess text data, including building a vocabulary from training data with a specified frequency threshold to manage rare words.
- **Model Training**: An HMM is trained using input text data, calculating transition and emission probabilities essential for sequence tagging.
- **Decoding**: Implementation of a Greedy Decoder and Viterbi algorithm to predict labels for new text sequences based on the trained HMM.
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
   ```
3. **Prepare Your Data**: Organize your data in the required format. The expected format is a tab-separated file where each line contains a sequence index, the token, and its corresponding tag. Sequences should be separated by a newline.
4. **Train the Model**:  Run the code to train the HMM on your training data. This process involves data loading, vocabulary building, and calculating the transition and emission probabilities.
5. **Decode Sequences**: Use the trained model to decode sequences in your test data. The greedy & viterbi decoder will output the predicted sequence of tags based on the model.
6. **Evaluate Performance**: Calculate the accuracy of the model on your development or test set to understand its performance.

## Data and Model Files

- **Training Data**: Text file with training sequences.
- **Test Data**: Text file with sequences for testing the model.
- **Vocabulary File**: Generated vocabulary with token counts.
- **Model File**:  JSON file storing the trained HMM's transition and emission probabilities.

## Usage Example:

After setting up your environment and data, you can train and evaluate the model as follows:

1. **Training**: Load your training data, build the vocabulary, and train the HMM.

```python
train_data, train_vocab = load_text_data('data/train', get_vocabulary=True)
train_data, vocab, vocab_size, total_words_before = build_vocabulary(train_data, train_vocab, threshold=2)
hmm = HiddenMarkovModel(vocab, train_data)
hmm.train()
hmm.save_model("data/hmm.json")
```

2. **Decoding and Evaluation**: Load your test data, decode the sequences using the Greedy Decoder, and evaluate the model's accuracy.

a. **Greedy**:
```python
test_data = load_text_data('data/test', get_vocabulary=False, separate_sentences=True, replace_unknown=True, vocab=hmm.lexicon)
test_data_orig = load_text_data('data/test', get_vocabulary=False, separate_sentences=True,  vocab=hmm.lexicon)
greedy_test = GreedyDecoder(test_data, hmm.labels, transition_prob, emission_prob, test_data_orig)
preds = greedy_test.decode()
acc = greedy_test.calculate_accuracy(greedy_test.get_targets())
print(f"Greedy Decoding Accuracy on test_data: {acc*100}")
```

   b. **Viterbi**:
   ```python
   viterbi_decoder = ViterbiDecoding(data, tag_list, transition_prob, emission_prob, data_orig)
   predictions = viterbi_decoder.predict()
   accuracy = viterbi_decoder.calculate_accuracy(viterbi_decoder.get_targets())
   print(f"Accuracy: {accuracy*100}%")
   ```

## Contributing
Feel free to fork the repository and submit pull requests.
