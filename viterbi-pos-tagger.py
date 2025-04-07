from collections import defaultdict, Counter

def read_pos_file(file_path):
    """
    Reads a POS-tagged file and returns a list of sentences,
    where each sentence is a list of (word, tag) tuples.
    """
    sentences = []
    current_sentence = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                word, tag = line.split()
                current_sentence.append((word, tag))
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
    return sentences

def calculate_prior_probabilities(training_data):
    """
    Calculates the prior probabilities for each POS tag pair (bigram)
    in the training data.
    """
    # Count the occurrences of each tag pair
    tag_bigrams = []
    for sentence in training_data:
        tags = [tag for _, tag in sentence]
        tag_bigrams.extend(list(zip(tags, tags[1:])))
    tag_bigram_counts = Counter(tag_bigrams)

    # Calculate total counts for each first tag in the bigram
    first_tag_counts = defaultdict(int)
    for (tag1, _), count in tag_bigram_counts.items():
        first_tag_counts[tag1] += count

    # Calculate prior probabilities
    prior_probabilities = {}
    for tag_bigram, count in tag_bigram_counts.items():
        tag1 = tag_bigram[0]
        prior_probabilities[tag_bigram] = count / first_tag_counts[tag1]

    return prior_probabilities

def calculate_likelihoods(training_data):
    """
    Calculates the likelihoods of each word given its POS tag in the training data.
    """
    # Count the occurrences of each word-tag pair
    word_tag_pairs = [(word, tag) for sentence in training_data for word, tag in sentence]
    word_tag_counts = Counter(word_tag_pairs)

    # Calculate total counts for each tag
    tag_counts = defaultdict(int)
    for (_, tag), count in word_tag_counts.items():
        tag_counts[tag] += count

    # Calculate likelihoods
    likelihoods = {}
    for word_tag, count in word_tag_counts.items():
        tag = word_tag[1]
        likelihoods[word_tag] = count / tag_counts[tag]

    return likelihoods

def get_tag_set(training_data):
    """
    Extracts the set of all possible tags from the training data.
    """
    tag_set = set()
    for sentence in training_data:
        for _, tag in sentence:
            tag_set.add(tag)
    return tag_set

def viterbi_algorithm(words, prior_probabilities, likelihoods, tag_set):
    """
    Implements the Viterbi algorithm for POS tagging.
    """
    # Initialize the dynamic programming table
    dp_table = [{} for _ in range(len(words))]
    backpointer = [{} for _ in range(len(words))]

    # Initialize the first column of the DP table
    for tag in tag_set:
        dp_table[0][tag] = prior_probabilities.get(('<s>', tag), 1e-6) * likelihoods.get((words[0], tag), 1e-6)
        backpointer[0][tag] = None

    # Fill in the rest of the DP table
    for i in range(1, len(words)):
        for current_tag in tag_set:
            max_prob = 0
            best_prev_tag = None
            for prev_tag in tag_set:
                prob = dp_table[i-1][prev_tag] * prior_probabilities.get((prev_tag, current_tag), 1e-6) * likelihoods.get((words[i], current_tag), 1e-6)
                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag
            dp_table[i][current_tag] = max_prob
            backpointer[i][current_tag] = best_prev_tag

    

    # Backtrack to find the best sequence of tags
    best_sequence = []
    max_final_prob = 0
    best_final_tag = None
    for tag in tag_set:
        if dp_table[-1][tag] > max_final_prob:
            max_final_prob = dp_table[-1][tag]
            best_final_tag = tag

    best_sequence.append(best_final_tag)
    for i in range(len(words) - 1, 0, -1):
        best_sequence.insert(0, backpointer[i][best_sequence[0]])

    return best_sequence

def tag_development_set(input_file, output_file, prior_probabilities, likelihoods, tag_set):
    """
    Tags the words in the development set using the Viterbi algorithm and writes the output to a file.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        words = []
        for line in infile:
            line = line.strip()
            if line:
                words.append(line)
            else:
                if words:
                    tags = viterbi_algorithm(words, prior_probabilities, likelihoods, tag_set)
                    for word, tag in zip(words, tags):
                        outfile.write(f"{word}\t{tag}\n")
                    outfile.write("\n")
                    words = []
            
            
def merge_corpora(training_file, development_file, combined_file):
    """
    Merges the training and development corpora into a single combined file.
    """
    with open(combined_file, 'w') as outfile:
        for fname in [training_file, development_file]:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

def main():
    # Read and process the original training data
    training_file = 'WSJ_02-21.pos'
    training_data = read_pos_file(training_file)
    prior_probabilities = calculate_prior_probabilities(training_data)
    likelihoods = calculate_likelihoods(training_data)
    tag_set = get_tag_set(training_data)

    # Merge the training and development corpora
    development_file = 'WSJ_24.pos'
    combined_file = 'WSJ_combined.pos'
    merge_corpora(training_file, development_file, combined_file)

    # Retrain the model on the combined corpus
    combined_training_data = read_pos_file(combined_file)
    prior_probabilities = calculate_prior_probabilities(combined_training_data)
    likelihoods = calculate_likelihoods(combined_training_data)

    # Tag the test set
    test_input_file = 'WSJ_23.words'
    test_output_file = 'submission.pos'
    tag_development_set(test_input_file, test_output_file, prior_probabilities, likelihoods, tag_set)

    print("Tagging of the test set is complete. Output is saved in", test_output_file)

if __name__ == '__main__':
    main()
