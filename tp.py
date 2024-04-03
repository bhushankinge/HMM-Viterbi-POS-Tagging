import json
import pandas as pd
from collections import Counter


def read_training_data(filepath):
    words = []  # Initialize an empty list to hold words
    pos_tags = []  # Initialize an empty list to hold POS tags
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                parts = line.split('\t')
                if len(parts) > 1:  # Ensure the line has enough parts
                    word = parts[1].strip()  # Extract the word
                    pos_tag = parts[2].strip()  # Extract the POS tag
                    words.append(word)  # Append the word to the list
                    pos_tags.append(pos_tag)  # Append the POS tag to the list
    return words, pos_tags

def count_words(words):
    word_counts = {}
    for word in words:
        # # Convert word to lowercase before counting
        # word_lower = word.lower()
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts

def store_unk_words(unk_words):
        with open("unk_words.txt", "w", encoding="utf-8") as file:
            for word in sorted(unk_words):
                file.write(f"{word}\n")


def create_vocabulary(word_counts, threshold=3):
    vocab = {'<unk>': 0}
    unk_words = set()
    for word, count in word_counts.items():
        if count < threshold:
            vocab['<unk>'] += count
            unk_words.add(word)
        else:
            vocab[word] = count
    store_unk_words(unk_words)
    return vocab

def output_vocabulary(vocab, filename):
    # Sort the vocabulary by count descending (excluding <unk>), then alphabetically for words with the same count
    sorted_vocab = sorted(vocab.items(), key=lambda x: (-x[1], x[0]) if x[0] != '<unk>' else (x[1], x[0]))

    with open(filename, 'w', encoding='utf-8') as file:
        # Initialize index counter, starting from 1 since <unk> will be 0
        index = 1  
        # First, write <unk> with index 0
        file.write(f"<unk>\t0\t{vocab['<unk>']}\n")
        
        # Write the rest of the vocabulary with incremented indices
        for word, count in sorted_vocab:
            if word != '<unk>':  # Skip <unk> since it's already written
                file.write(f"{word}\t{index}\t{count}\n")
                index += 1

def read_unk_words(filename="unk_words.txt"):
    with open(filename, "r", encoding="utf-8") as file:
        return set(word.strip() for word in file.readlines())

def parse_training_data(filename, unk_words):
    transitions, emissions = [], []
    prev_tag = '<START_TAG>'

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                transitions.append((prev_tag, "<END_TAG>"))
                prev_tag = '<START_TAG>'
                continue
            
            parts = line.strip().split('\t')
            if len(parts) == 3:
                index, word, tag = parts
                # word = word.lower()  # Ensure lowercase to match unk_words
                if word in unk_words:
                    word = "<unk>"
                
                transitions.append((prev_tag, tag))
                emissions.append((tag, word))
                prev_tag = tag
        
        # Handle the transition for the last sentence
        transitions.append((prev_tag, "<END_TAG>"))

    return transitions, emissions

def count_transitions_emissions(transitions, emissions):
    from collections import Counter
    transition_counts = Counter(transitions)
    emission_counts = Counter(emissions)
    return transition_counts, emission_counts

def calculate_probabilities(transition_counts, emission_counts):
    tag_counts = Counter()
    start_tag_counts = Counter()  # New counter for <start> transitions

    # Counting transitions from <start> separately
    for (prev_tag, tag), count in transition_counts.items():
        if prev_tag == "<START_TAG>":
            start_tag_counts[tag] += count
        else:
            tag_counts[prev_tag] += count

    for (tag, _), count in emission_counts.items():
        tag_counts[tag] += count

    # Calculating probabilities including <start>
    transition_probs = {}
    for (prev_tag, tag), count in transition_counts.items():
        if prev_tag == "<START_TAG>":
            transition_probs[(prev_tag, tag)] = count / sum(start_tag_counts.values())
        elif prev_tag not in ["<END_TAG>"]:
            transition_probs[(prev_tag, tag)] = count / tag_counts[prev_tag]

    emission_probs = {k: v / tag_counts[k[0]] for k, v in emission_counts.items()}
    
    return transition_probs, emission_probs


def output_model(transition_probs, emission_probs):
    # Convert tuple keys to string format
    transition_probs_str = {f"{k[0]}|{k[1]}": v for k, v in transition_probs.items()}
    emission_probs_str = {f"{k[0]}|{k[1]}": v for k, v in emission_probs.items()}

    model = {
        "transition": transition_probs_str,
        "emission": emission_probs_str
    }

    with open("hmm.json", "w", encoding="utf-8") as file:
        json.dump(model, file, indent=4)

        
def load_model(model_path):
    with open(model_path, 'r') as file:
        model = json.load(file)
    return model['transition'], model['emission']


def load_vocab(vocab_path):
    vocab = set()
    with open(vocab_path, 'r') as file:
        for line in file:
            word = line.split('\t')[0]
            vocab.add(word)
    return vocab

def process_data(data_path, vocab):
    sentences = []
    with open(data_path, 'r') as file:
        sentence = []
        for line in file:
            if line.strip():
                word = line.split('\t')[1]
                sentence.append(word)
            else:
                sentences.append(sentence)
                sentence = []
    if sentence: sentences.append(sentence)
    return sentences

def write_output(sentences, tags, output_path):
    with open(output_path, 'w') as file:
        for sentence, tag_seq in zip(sentences, tags):
            for word, tag in zip(sentence, tag_seq):
                file.write(f"{word}\t{tag}\n")
            file.write("\n")

#####################################################################################
            
import json

# Function to perform greedy decoding for a list of words
def greedy_decode(words, transition, emission):
    prev_tag = "<START_TAG>"
    word_tags = []
    for word in words:
        max_prob = 0
        best_tag = None
        for candidate_tag, trans_prob in transition.items():
            prev_tag_candidate, current_tag = candidate_tag.split('|')
            if prev_tag_candidate == prev_tag:
                em_key = f"{current_tag}|{word}"
                em_prob = emission.get(em_key, emission.get(f"{current_tag}|<unk>", 0))
                prob = trans_prob * em_prob
                if prob > max_prob:
                    max_prob = prob
                    best_tag = current_tag
        word_tags.append((word, best_tag))
        prev_tag = best_tag
    return word_tags

#####################################################################################
import pandas as pd

def split_into_dfs(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
    
    records = content.split('\n\n')
    
    dfs = []
    for record in records:
        lines = record.split('\n')
        rows = [line.split('\t') for line in lines]
        df = pd.DataFrame(rows, columns=['Index', 'Word', 'POS'])
        dfs.append(df)
    
    return dfs

def get_words_and_pos_from_dfs(dfs):
    words = []
    pos_tags = []
    for df in dfs:
        words.extend(df['Word'].tolist())
        pos_tags.extend(df['POS'].tolist())
    return words, pos_tags


dfs_list = split_into_dfs('data/dev')
words_dev, pos_tag_dev = get_words_and_pos_from_dfs(dfs_list)


words_test = []

# Replace 'data/test' with the correct path to your test file
with open('data/test', 'r') as file:
    for line in file:
        # Strip leading and trailing whitespace and split the line
        parts = line.strip().split(maxsplit=1)
        # Only proceed if the line was successfully split into two parts
        if len(parts) == 2:
            _, word = parts
            words_test.append(word)


if __name__ == "__main__":
    # Your existing code for reading and processing vocabulary
    words, pos_tags = read_training_data("./data/train")  # Adjust filepath as necessary
    word_counts = count_words(words)
    vocab = create_vocabulary(word_counts)
    unk_count = vocab['<unk>']
    output_vocabulary(vocab, "vocab.txt")
    print(f"Threshold for unknown words: 3")
    print(f"Total size of the vocabulary: {len(vocab) + 1}")
    print(f"Total occurrences of '<unk>': {unk_count}")


    # New code for HMM model learning
    unk_words = read_unk_words()  # Read the unknown words
    transitions, emissions = parse_training_data("./data/train", unk_words)
    transition_counts, emission_counts = count_transitions_emissions(transitions, emissions)
    transition_probs, emission_probs = calculate_probabilities(transition_counts, emission_counts)
    output_model(transition_probs, emission_probs)
    
    # Outputs to review model complexity
    print(f"Number of transition parameters: {len(transition_probs)}")
    print(f"Number of emission parameters: {len(emission_probs)}")

    transition_probs, emission_probs = load_model("hmm.json")
    vocab = load_vocab("vocab.txt")
    
    decoded_tags = greedy_decode(words_dev, transition_probs, emission_probs)

    #get the decoded_tags for the test data
    decoded_tags_test = greedy_decode(words_test, transition_probs, emission_probs )

    with open("greedy.out", "w") as f:
        index = 0;
        for word, tag in decoded_tags_test:
            f.write(f"{index}\t{word}\t{tag}\n")
            index +=1

    predicted_tags = [tag for word, tag in decoded_tags]
    correct_predictions = sum(p == a for p, a in zip(predicted_tags, pos_tag_dev))

    accuracy = correct_predictions / len(pos_tag_dev) if pos_tag_dev else 0

    print(f"Accuracy: {accuracy:.4f}")

        