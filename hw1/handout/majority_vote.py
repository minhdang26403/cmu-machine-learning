import csv
import sys
from collections import Counter


def find_majority_label(attribute_count):
    majority_label = ""
    majority_label_count = 0
    for label, count in attribute_count.items():
        if count > majority_label_count or (
            count == majority_label_count and label > majority_label
        ):
            majority_label_count = count
            majority_label = label
    return majority_label


def majority_vote():
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]

    freq_count = Counter()
    num_train_samples = 0
    with open(train_input, "r") as f:
        # Create a CSV reader object with tab delimiter for TSV files.
        reader = csv.reader(f, delimiter="\t")

        # Skip the header row by advancing the iterator. Pass None as default value
        # to avoid StopIteration error.
        next(reader, None)

        for row in reader:
            label = row[-1]
            freq_count[label] += 1
            num_train_samples += 1

    # Find the label with the highest frequency
    majority_label = find_majority_label(freq_count)

    # Given the majority label, count the number of mismatch in test data
    num_test_samples = 0
    num_test_error_count = 0
    with open(test_input, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            label = row[-1]
            if label != majority_label:
                num_test_error_count += 1
            num_test_samples += 1

    with open(train_out, "w") as f:
        result = (majority_label + "\n") * num_train_samples
        f.write(result)

    with open(test_out, "w") as f:
        result = (majority_label + "\n") * num_test_samples
        f.write(result)

    # Calculate train error rate and test error rate
    train_error = 1 - freq_count[majority_label] / num_train_samples
    test_error = num_test_error_count / num_test_samples

    # Write the metrics to the metrics_out file
    with open(metrics_out, "w") as f:
        f.write(f"error(train): {train_error:.6f}\n")
        f.write(f"error(test): {test_error:.6f}")


if __name__ == "__main__":
    majority_vote()
