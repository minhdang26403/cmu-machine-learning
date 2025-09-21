import argparse

import matplotlib.pyplot as plt

from neuralnet import NN, args2data, random_init, zero_init

# This takes care of command line argument parsing for you!
# To access a specific argument, simply access args.<argument name>.
parser = argparse.ArgumentParser()
parser.add_argument("train_input", type=str, help="path to training input .csv file")
parser.add_argument(
    "validation_input", type=str, help="path to validation input .csv file"
)
parser.add_argument(
    "train_out", type=str, help="path to store prediction on training data"
)
parser.add_argument(
    "validation_out", type=str, help="path to store prediction on validation data"
)
parser.add_argument(
    "metrics_out", type=str, help="path to store training and testing metrics"
)
parser.add_argument("num_epoch", type=int, help="number of training epochs")
parser.add_argument("hidden_units", type=int, help="number of hidden units")
parser.add_argument(
    "init_flag",
    type=int,
    choices=[1, 2],
    help="weight initialization functions, 1: random",
)
parser.add_argument("learning_rate", type=float, help="learning rate")


def plot_losses_vs_epochs(
    epochs_values: list[int],
    train_losses: list[float],
    val_losses: list[float],
    learning_rate: float,
    title_suffix: str = "",
    xlabel: str = "Epoch Number",
    ylabel: str = "Average Cross-Entropy Loss",
):
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs_values, train_losses, marker="o", markersize=4, label="Training Loss"
    )
    plt.plot(
        epochs_values, val_losses, marker="x", markersize=4, label="Validation Loss"
    )

    plt.title(f"Cross-Entropy Loss vs. Epochs (LR: {learning_rate}){title_suffix}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"losses_vs_epochs_lr_{str(learning_rate).replace('.', '_')}.png")
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    # Define our labels
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]

    args = parser.parse_args()
    # Call args2data to get all data + argument values
    # See the docstring of `args2data` for an explanation of
    # what is being returned.
    (
        X_tr,
        y_tr,
        X_test,
        y_test,
        out_tr,
        out_te,
        out_metrics,
        n_epochs,
        n_hid,
        init_flag,
        lr,
    ) = args2data(args)

    # Define experiment parameters
    learning_rates = [0.03, 0.003, 0.0003]
    epochs_values = list(range(1, n_epochs + 1))

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        nn = NN(
            input_size=X_tr.shape[-1],
            hidden_size=n_hid,
            output_size=len(labels),
            weight_init_fn=zero_init if init_flag == 2 else random_init,
            learning_rate=lr,
        )

        train_losses, val_losses = nn.train(X_tr, y_tr, X_test, y_test, n_epochs)

        # Plot for the current learning rate
        plot_losses_vs_epochs(epochs_values, train_losses, val_losses, lr)
        print(
            f"Plot saved for LR={lr} as losses_vs_epochs_lr_{str(lr).replace('.', '_')}.png"
        )

    print("All plots generated successfully.")
