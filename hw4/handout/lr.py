import argparse

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x: np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta: np.ndarray,  # shape (D,) where D is feature dim
    X: np.ndarray,  # shape (N, D) where N is num of examples
    y: np.ndarray,  # shape (N,)
    X_val: np.ndarray,  # shape (N_val, D) where N_val is num of validation examples
    y_val: np.ndarray,  # shape (N_val,)
    num_epoch: int,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_nlls = np.zeros(num_epoch)
    val_nlls = np.zeros(num_epoch)
    for epoch in range(num_epoch):
        for i in range(X.shape[0]):
            # Single example prediction and gradient
            prediction = sigmoid(X[i] @ theta)
            gradient = (prediction - y[i]) * X[i]
            theta -= learning_rate * gradient
        # Record NLL after each epoch
        train_nlls[epoch] = negative_log_likelihood(theta, X, y)
        val_nlls[epoch] = negative_log_likelihood(theta, X_val, y_val)
    return train_nlls, val_nlls


def predict(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return sigmoid(X @ theta) > 0.5


def compute_error(y_pred: np.ndarray, y: np.ndarray) -> float:
    return np.mean(np.abs(y_pred - y))


def negative_log_likelihood(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    # Clip probabilities to avoid log(0)
    predictions = sigmoid(X @ theta)
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
    # Compute negative log-likelihood
    nll = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return nll


def load_formatted_dataset(file):
    """
    Loads formatted feature dataset with format: label\tfeat1\tfeat2\t...\tfeat300
    Returns: labels, X (with bias column added)
    """
    # Load all columns as floats
    dataset = np.loadtxt(file, delimiter="\t", dtype=float)
    labels = dataset[:, 0]  # First column is labels
    features = dataset[:, 1:]  # Remaining columns are features
    # Add bias column (column of ones)
    X = np.column_stack((np.ones(len(dataset)), features))
    return labels, X


def save_predictions(predictions, file):
    np.savetxt(file, predictions, delimiter="\t", fmt="%d")


def plot_nll(train_nlls, val_nlls, num_epoch):
    epochs = range(1, num_epoch + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_nlls, label="Training NLL")
    plt.plot(epochs, val_nlls, label="Validation NLL")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Negative Log-Likelihood vs. Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("nll_plot.png")


def plot_average_nll(train_X, train_labels, validation_X, validation_labels, num_epoch):
    train_nlls_list = []
    learning_rates = [0.1, 0.01, 0.001]
    for learning_rate in learning_rates:
        theta = np.zeros(train_X.shape[1])
        train_nlls, _ = train(
            theta,
            train_X,
            train_labels,
            validation_X,
            validation_labels,
            num_epoch,
            learning_rate,
        )
        train_nlls_list.append(train_nlls)

    epochs = range(1, num_epoch + 1)
    plt.figure(figsize=(10, 6))
    for train_nlls, learning_rate in zip(train_nlls_list, learning_rates):
        plt.plot(epochs, train_nlls, label=f"Training NLL (LR={learning_rate})")
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Negative Log-Likelihood vs. Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("average_nll_plot.png")


if __name__ == "__main__":
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help="path to formatted training data")
    parser.add_argument(
        "validation_input", type=str, help="path to formatted validation data"
    )
    parser.add_argument("test_input", type=str, help="path to formatted test data")
    parser.add_argument(
        "train_out", type=str, help="file to write train predictions to"
    )
    parser.add_argument("test_out", type=str, help="file to write test predictions to")
    parser.add_argument("metrics_out", type=str, help="file to write metrics to")
    parser.add_argument(
        "num_epoch",
        type=int,
        help="number of epochs of stochastic gradient descent to run",
    )
    parser.add_argument(
        "learning_rate",
        type=float,
        help="learning rate for stochastic gradient descent",
    )
    args = parser.parse_args()

    train_labels, train_X = load_formatted_dataset(args.train_input)
    validation_labels, validation_X = load_formatted_dataset(args.validation_input)
    test_labels, test_X = load_formatted_dataset(args.test_input)

    theta = np.zeros(train_X.shape[1])

    train_nlls, val_nlls = train(
        theta,
        train_X,
        train_labels,
        validation_X,
        validation_labels,
        args.num_epoch,
        args.learning_rate,
    )

    train_predictions = predict(theta, train_X)
    test_predictions = predict(theta, test_X)

    np.savetxt(args.train_out, train_predictions, delimiter="\t", fmt="%d")
    np.savetxt(args.test_out, test_predictions, delimiter="\t", fmt="%d")

    # Save metrics with labels in the expected format
    train_error = compute_error(train_predictions, train_labels)
    test_error = compute_error(test_predictions, test_labels)

    with open(args.metrics_out, "w") as f:
        f.write(f"error(train): {train_error:.6f}\n")
        f.write(f"error(test): {test_error:.6f}")

    plot_nll(train_nlls, val_nlls, args.num_epoch)
    plot_average_nll(
        train_X, train_labels, validation_X, validation_labels, args.num_epoch
    )
