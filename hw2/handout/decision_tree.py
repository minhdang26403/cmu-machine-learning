import argparse
import csv
from collections import Counter

from ml_utils import calculate_entropy, find_majority_label


class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree.
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired
    """

    def __init__(self, attr=None, vote=None, label_counts=None):
        """
        Initialize a Node in the decision tree.

        Args:
            attr: The attribute used for splitting at this node (None for leaf nodes)
            vote: The majority class label at this node (used for leaf nodes)
            label_counts: Counter of label frequencies for the data at this node
        """
        self.left = None
        self.right = None
        self.attr = attr
        self.vote = vote
        self.label_counts = label_counts


def calculate_mutual_information(
    label_counts, attribute_value_label_counts, total_samples
):
    """
    Calculate mutual information I(Y; X) = H(Y) - H(Y|X)

    Args:
        label_counts: Counter of label frequencies in the entire dataset
        attribute_value_label_counts: Dict where key is attribute value (e.g., 0, 1)
                                      and value is Counter of label frequencies for that
                                      attribute value
        total_samples: Total number of samples in the dataset

    Returns:
        Mutual information value
    """
    # Calculate H(Y) - entropy of labels
    label_probabilities = [count / total_samples for count in label_counts.values()]
    label_entropy = calculate_entropy(label_probabilities)

    # Calculate H(Y|X) - conditional entropy
    conditional_entropy = 0.0

    for attr_value, attr_label_counts in attribute_value_label_counts.items():
        # P(X = attr_value)
        attr_count = sum(attr_label_counts.values())
        attr_probability = attr_count / total_samples

        # H(Y|X = attr_value) - entropy of labels given this attribute value
        if attr_count > 0:  # Avoid division by zero
            conditional_probabilities = [
                attr_label_counts.get(label, 0) / attr_count
                for label in label_counts.keys()
            ]
            attr_conditional_entropy = calculate_entropy(conditional_probabilities)
            conditional_entropy += attr_probability * attr_conditional_entropy

    # I(Y;X) = H(Y) - H(Y|X)
    mutual_information = label_entropy - conditional_entropy
    return mutual_information


def calculate_mutual_information_for_attribute(data, attribute_index):
    """
    Calculate mutual information for a specific attribute in the dataset.

    Args:
        data: List of lists where each inner list is [feature1, feature2, ..., label]
        attribute_index: Index of the attribute to calculate mutual information for

    Returns:
        Mutual information value for the attribute
    """
    if not data:
        return 0.0

    num_samples = len(data)
    label_counts = Counter(row[-1] for row in data)

    # Count labels for each value of the attribute
    attribute_value_label_counts = {}
    for row in data:
        attr_value = row[attribute_index]
        label = row[-1]

        if attr_value not in attribute_value_label_counts:
            attribute_value_label_counts[attr_value] = Counter()

        attribute_value_label_counts[attr_value][label] += 1

    return calculate_mutual_information(
        label_counts, attribute_value_label_counts, num_samples
    )


def find_best_attribute(data, attributes, attribute_to_index):
    """
    Find the best attribute to split on based on mutual information.

    Args:
        data: 2D list where each row is [feature1, feature2, ..., label]
        attributes: List of remaining attribute names
        attribute_to_index: Dict mapping attribute names to their column indices in data

    Returns:
        tuple: (best_attribute_name, best_attribute_data_index, best_mutual_information)
    """
    best_attribute_name = None
    best_attribute_data_index = -1
    best_mutual_information = 0

    for attr_name in attributes:
        data_index = attribute_to_index[attr_name]
        mutual_information = calculate_mutual_information_for_attribute(
            data, data_index
        )
        if mutual_information > best_mutual_information:
            best_mutual_information = mutual_information
            best_attribute_name = attr_name
            best_attribute_data_index = data_index

    return best_attribute_name, best_attribute_data_index, best_mutual_information


def build_decision_tree(data, attributes, attribute_to_index, max_depth):
    """
    Recursively build a decision tree using mutual information.

    Args:
        data: 2D list where each row is [feature1, feature2, ..., label]
        attributes: List of remaining attribute names
        attribute_to_index: Dict mapping attribute names to their column indices in data
        max_depth: Maximum depth of the tree

    Returns:
        Node: Root node of the decision tree
    """
    label_counts = Counter(row[-1] for row in data)
    majority_label = find_majority_label(label_counts)

    # Base cases: max depth reached, no attributes left, or all labels are the same
    if max_depth == 0 or len(attributes) == 0 or len(label_counts) == 1:
        return Node(vote=majority_label, label_counts=label_counts)

    best_attribute_name, best_attribute_data_index, best_mutual_information = (
        find_best_attribute(data, attributes, attribute_to_index)
    )

    # If no attribute provides information gain, create a leaf node
    if best_mutual_information == 0:
        return Node(vote=majority_label, label_counts=label_counts)

    # Create internal node
    root = Node(attr=best_attribute_name, label_counts=label_counts)

    # Remove the best attribute from the list
    attributes.remove(best_attribute_name)

    # Split data based on the best attribute using its correct data index
    left_data = [row for row in data if row[best_attribute_data_index] == "0"]
    right_data = [row for row in data if row[best_attribute_data_index] == "1"]

    # Recursively build left and right subtrees
    root.left = build_decision_tree(
        left_data, attributes, attribute_to_index, max_depth - 1
    )
    root.right = build_decision_tree(
        right_data, attributes, attribute_to_index, max_depth - 1
    )

    # Recover the attributes set for the caller
    attributes.add(best_attribute_name)

    return root


def print_tree(node, depth, attribute, label, file):
    """
    Recursively print the decision tree structure to a file.

    Args:
        node: Current node to print
        depth: Current depth in the tree (for indentation)
        attribute: Parent attribute name (for branch labels)
        label: Branch value from parent ("0" or "1")
        file: File object to write to

    The output format shows:
    - Indentation with "| " for each level
    - Branch conditions like "attribute = value: "
    - Label counts like "[count_0 0/count_1 1]"
    """
    if node is None:
        return

    line = ""
    for _ in range(depth):
        line += "| "

    if attribute is not None:
        assert label in ["0", "1"]
        line += f"{attribute} = {label}: "

    line += f"[{node.label_counts['0']} 0/{node.label_counts['1']} 1]\n"
    file.write(line)

    print_tree(node.left, depth + 1, node.attr, "0", file)
    print_tree(node.right, depth + 1, node.attr, "1", file)


def predict(node: Node, sample: dict[str, str]) -> str:
    """
    Make a prediction for a sample using the trained decision tree.

    Args:
        node: Current node in the decision tree
        sample: Dictionary mapping attribute names to their values

    Returns:
        str: Predicted class label ("0" or "1")

    The function traverses the tree by following the path determined by
    the sample's attribute values until reaching a leaf node.
    """
    if node.attr is None:
        return node.vote
    else:
        if sample[node.attr] == "0":
            return predict(node.left, sample)
        else:
            return predict(node.right, sample)


if __name__ == "__main__":
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_input", type=str, help="path to training input .tsv file"
    )
    parser.add_argument("test_input", type=str, help="path to the test input .tsv file")
    parser.add_argument(
        "max_depth", type=int, help="maximum depth to which the tree should be built"
    )
    parser.add_argument(
        "train_out",
        type=str,
        help="path to output .txt file to which the feature extractions on the "
        "training data should be written",
    )
    parser.add_argument(
        "test_out",
        type=str,
        help="path to output .txt file to which the feature extractions on the test "
        "data should be written",
    )
    parser.add_argument(
        "metrics_out",
        type=str,
        help="path of the output .txt file to which metrics such as train and test "
        "error should be written",
    )
    parser.add_argument(
        "print_out",
        type=str,
        help="path of the output .txt file to which the printed tree should be written",
    )
    args = parser.parse_args()

    # Read the first line to get the list of attributes
    with open(args.train_input, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        all_columns = next(reader)  # First line contains all column names
        attributes = all_columns[:-1]  # Exclude the label column (last column)

        # Create mapping from attribute names to their column indices in data
        attribute_to_index = {attr: idx for idx, attr in enumerate(attributes)}

        train_data = list(reader)

    # Build the decision tree
    print(f"Building decision tree with max_depth={args.max_depth}")
    print(f"Attributes: {attributes}")
    print(f"Training data shape: {len(train_data)} samples, {len(attributes)} features")

    decision_tree = build_decision_tree(
        train_data, set(attributes), attribute_to_index, args.max_depth
    )

    print("Decision tree built successfully!")

    train_error = 0
    test_error = 0

    with open(args.train_out, "w") as f:
        for row in train_data:
            sample = {attributes[i]: row[i] for i in range(len(attributes))}
            prediction = predict(decision_tree, sample)
            if prediction != row[-1]:
                train_error += 1
            f.write(f"{prediction}\n")

    with open(args.test_input, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        test_data = list(reader)

    with open(args.test_out, "w") as f:
        for row in test_data:
            sample = {attributes[i]: row[i] for i in range(len(attributes))}
            prediction = predict(decision_tree, sample)
            if prediction != row[-1]:
                test_error += 1
            f.write(f"{prediction}\n")

    with open(args.metrics_out, "w") as f:
        f.write(f"error(train): {train_error / len(train_data)}\n")
        f.write(f"error(test): {test_error / len(test_data)}\n")

    # Here is a recommended way to print the tree to a file
    with open(args.print_out, "w") as file:
        print_tree(decision_tree, 0, None, "?", file)
