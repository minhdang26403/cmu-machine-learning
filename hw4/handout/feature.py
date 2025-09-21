import argparse
import csv

import numpy as np

VECTOR_LEN = 300  # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(
        file, delimiter="\t", comments=None, encoding="utf-8", dtype="l,O"
    )
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = {}
    with open(file, encoding="utf-8") as f:
        read_file = csv.reader(f, delimiter="\t")
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map


def get_feature_vector(dataset, glow_map):
    """
    Takes in a sentence and returns feature vectors and labels separately.

    Returns:
        features: np.ndarray of shape (N, VECTOR_LEN) - feature vectors
        labels: np.ndarray of shape (N,) - corresponding labels
    """
    N = len(dataset)
    features = np.zeros((N, VECTOR_LEN))
    labels = np.zeros(N, dtype=int)

    for i, (label, review) in enumerate(dataset):
        labels[i] = label
        words = review.split()
        valid_words = [word for word in words if word in glow_map]

        if len(valid_words) > 0:
            # Compute mean of valid word embeddings
            word_vectors = np.array([glow_map[word] for word in valid_words])
            features[i] = np.mean(word_vectors, axis=0)

    return features, labels


def save_feature_vector(dataset, glow_map, out_file):
    features, labels = get_feature_vector(dataset, glow_map)

    fmt_parts = ["%.6f"] * (VECTOR_LEN + 1)
    fmt_string = "\t".join(fmt_parts)

    np.savetxt(
        out_file,
        np.column_stack((labels.astype(float), features)),
        delimiter="",
        fmt=fmt_string,
    )


if __name__ == "__main__":
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_input", type=str, help="path to training input .tsv file"
    )
    parser.add_argument(
        "validation_input", type=str, help="path to validation input .tsv file"
    )
    parser.add_argument("test_input", type=str, help="path to the input .tsv file")
    parser.add_argument(
        "feature_dictionary_in",
        type=str,
        help="path to the GloVe feature dictionary .txt file",
    )
    parser.add_argument(
        "train_out",
        type=str,
        help="path to output .tsv file to which the feature extractions on the training"
        "data should be written",
    )
    parser.add_argument(
        "validation_out",
        type=str,
        help="path to output .tsv file to which the feature extractions on the"
        "validation data should be written",
    )
    parser.add_argument(
        "test_out",
        type=str,
        help="path to output .tsv file to which the feature extractions on the test"
        "data should be written",
    )
    args = parser.parse_args()

    train_dataset = load_tsv_dataset(args.train_input)
    validation_dataset = load_tsv_dataset(args.validation_input)
    test_dataset = load_tsv_dataset(args.test_input)
    glow_map = load_feature_dictionary(args.feature_dictionary_in)

    save_feature_vector(train_dataset, glow_map, args.train_out)
    save_feature_vector(validation_dataset, glow_map, args.validation_out)
    save_feature_vector(test_dataset, glow_map, args.test_out)
