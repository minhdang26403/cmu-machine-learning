import csv
import sys
from collections import Counter

from ml_utils import calculate_entropy, find_majority_label


def inspect():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    label_counts = Counter()
    total_samples = 0

    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            label = row[-1]
            label_counts[label] += 1
            total_samples += 1

    probabilities = [count / total_samples for count in label_counts.values()]
    entropy = calculate_entropy(probabilities)

    majority_label = find_majority_label(label_counts)
    error = 1 - label_counts[majority_label] / total_samples

    with open(output_file, "w") as f:
        f.write(f"entropy: {entropy}\n")
        f.write(f"error: {error}\n")

    print(f"entropy: {entropy}")
    print(f"error: {error}")


if __name__ == "__main__":
    inspect()
