# open csv file in latex_template/fairness_dataset.csv
# the last column is the label
# ignore the first column
# take the average of the remaining columns
# if the average is greater than 198.09, then the label is 1, otherwise 0
# print the error rate

import csv


def main():
    data = []
    labels = []
    with open("latex_template/fairness_dataset.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first empty line
        next(reader)  # Skip the header row
        for row in reader:
            # Skip the first column (Region)
            # The last column is the label
            # Take the average of the remaining columns
            if row[0] == "A":
                continue
            numeric_values = [float(x) for x in row[1:-1]]
            data.append(sum(numeric_values) / len(numeric_values))
            labels.append(int(row[-1]))

    n = len(labels)

    # if the average is greater than 198.09, then the label is 1, otherwise 0
    predictions = [1 if avg > 198.09 else 0 for avg in data]

    assert n == len(predictions)

    # print the error rate
    correct_predictions = sum(1 for i in range(n) if labels[i] == predictions[i])

    # print the error rate
    error_rate = 1 - (correct_predictions / n)
    print(f"Correct predictions: {correct_predictions}")
    print(f"Total predictions: {n}")
    print(f"Error rate: {error_rate:.2f}")

    false_positives = sum(1 for i in range(n) if labels[i] == 0 and predictions[i] == 1)
    false_negatives = sum(1 for i in range(n) if labels[i] == 1 and predictions[i] == 0)
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")

    print(f"Positive rate: {sum(predictions) / n}")

    print(f"Accuracy: {correct_predictions / n}")

    total_positives = sum(labels)
    total_negatives = n - total_positives

    false_positive_rate = false_positives / total_negatives
    false_negative_rate = false_negatives / total_positives
    print(f"False positive rate: {false_positive_rate}")
    print(f"False negative rate: {false_negative_rate}")

    print(f"FPR / FNR ratio: {false_positive_rate / false_negative_rate:.3f}")

    true_positives = sum(1 for i in range(n) if labels[i] == 1 and predictions[i] == 1)
    true_negatives = sum(1 for i in range(n) if labels[i] == 0 and predictions[i] == 0)
    positive_predictive_value = true_positives / (true_positives + false_positives)
    negative_predictive_value = true_negatives / (true_negatives + false_negatives)
    print(f"Positive predictive value: {positive_predictive_value}")
    print(f"Negative predictive value: {negative_predictive_value}")

    print(
        f"PPV / NPV ratio: {positive_predictive_value / negative_predictive_value:.3f}"
    )


if __name__ == "__main__":
    main()
