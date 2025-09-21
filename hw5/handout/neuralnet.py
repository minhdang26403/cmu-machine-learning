"""
neuralnet.py

What you need to do:
- Complete random_init
- Implement SoftMaxCrossEntropy methods
- Implement Sigmoid methods
- Implement Linear methods
- Implement NN methods

It is ***strongly advised*** that you finish the Written portion -- at the
very least, problems 1 and 2 -- before you attempt this programming
assignment; the code for forward and backprop relies heavily on the formulas
you derive in those problems.

Sidenote: We annotate our functions and methods with type hints, which
specify the types of the parameters and the returns. For more on the type
hinting syntax, see https://docs.python.org/3/library/typing.html.
"""

import argparse
from typing import Callable, List, Tuple

import numpy as np

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


def args2data(
    args,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, str, int, int, int, float
]:
    """
    DO NOT modify this function.

    Parse command line arguments, create train/test data and labels.
    :return:
    X_tr: train data *without label column and without bias folded in
        (numpy array)
    y_tr: train label (numpy array)
    X_te: test data *without label column and without bias folded in*
        (numpy array)
    y_te: test label (numpy array)
    out_tr: file for predicted output for train data (file)
    out_te: file for predicted output for test data (file)
    out_metrics: file for output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """
    # Get data from arguments
    out_tr = args.train_out
    out_te = args.validation_out
    out_metrics = args.metrics_out
    n_epochs = args.num_epoch
    n_hid = args.hidden_units
    init_flag = args.init_flag
    lr = args.learning_rate

    X_tr = np.loadtxt(args.train_input, delimiter=",")
    y_tr = X_tr[:, 0].astype(int)
    X_tr = X_tr[:, 1:]  # cut off label column

    X_te = np.loadtxt(args.validation_input, delimiter=",")
    y_te = X_te[:, 0].astype(int)
    X_te = X_te[:, 1:]  # cut off label column

    return (
        X_tr,
        y_tr,
        X_te,
        y_te,
        out_tr,
        out_te,
        out_metrics,
        n_epochs,
        n_hid,
        init_flag,
        lr,
    )


def shuffle(X, y, epoch):
    """
    DO NOT modify this function.

    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def zero_init(shape):
    """
    DO NOT modify this function.

    ZERO Initialization: All weights are initialized to 0.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape=shape)


def random_init(shape):
    """

    RANDOM Initialization: The weights are initialized randomly from a uniform
        distribution from -0.1 to 0.1.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    M, D = shape
    np.random.seed(M * D)  # Don't change this line!

    # TODO: create the random matrix here!
    # Hint: numpy might have some useful function for this
    # Hint: make sure you have the right distribution
    return np.random.uniform(-0.1, 0.1, shape)


class SoftMaxCrossEntropy:
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Implement softmax function.
        :param z: input logits of shape (num_classes,)
        :return: softmax output of shape (num_classes,)
        """
        # TODO: implement
        return np.exp(z) / np.sum(np.exp(z))

    def _cross_entropy(self, y: int, y_hat: np.ndarray) -> float:
        """
        Compute cross entropy loss.
        :param y: integer class label
        :param y_hat: prediction with shape (num_classes,)
        :return: cross entropy loss
        """
        # TODO: implement
        return -np.log(y_hat[y])

    def forward(self, z: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Compute softmax and cross entropy loss.
        :param z: input logits of shape (num_classes,)
        :param y: integer class label
        :return:
            y: predictions from softmax as an np.ndarray
            loss: cross entropy loss
        """
        # TODO: Call your implementations of _softmax and _cross_entropy here
        y_hat = self._softmax(z)
        loss = self._cross_entropy(y, y_hat)
        return y_hat, loss

    def backward(self, y: int, y_hat: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. ** softmax input **.
        Note that here instead of calculating the gradient w.r.t. the softmax
        probabilities, we are directly computing gradient w.r.t. the softmax
        input.

        Try deriving the gradient yourself (see Question 1.2(b) on the written),
        and you'll see why we want to calculate this in a single step.

        :param y: integer class label
        :param y_hat: predicted softmax probability with shape (num_classes,)
        :return: gradient with shape (num_classes,)
        """
        # TODO: implement using the formula you derived in the written
        grad = y_hat.copy()
        grad[y] -= 1

        return grad


class Sigmoid:
    def __init__(self):
        """
        Initialize state for sigmoid activation layer
        """
        # TODO Initialize any additional values you may need to store for the
        #  backward pass here
        self.z = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Take sigmoid of input x.
        :param x: Input to activation function (i.e. output of the previous
                  linear layer), with shape (output_size,)
        :return: Output of sigmoid activation function with shape
            (output_size,)
        """
        # TODO: perform forward pass and save any values you may need for
        #  the backward pass
        self.z = 1 / (1 + np.exp(-x))
        return self.z

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output of
            sigmoid activation
        :return: partial derivative of loss with respect to input of
            sigmoid activation
        """
        # TODO: implement
        return dz * self.z * (1 - self.z)


# This refers to a function type that takes in a tuple of 2 integers (row, col)
# and returns a numpy array (which should have the specified dimensions).
INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]


class Linear:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_init_fn: INIT_FN_TYPE,
        learning_rate: float,
    ):
        """
        :param input_size: number of units in the input of the layer
                           *not including* the folded bias
        :param output_size: number of units in the output of the layer
        :param weight_init_fn: function that creates and initializes weight
                               matrices for layer. This function takes in a
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        # Initialize learning rate for SGD
        self.lr = learning_rate

        # TODO: Initialize weight matrix for this layer - since we are
        #  folding the bias into the weight matrix, be careful about the
        #  shape you pass in.
        #  To be consistent with the formulas you derived in the written and
        #  in order for the unit tests to work correctly,
        #  the first dimension should be the output size
        self.w = weight_init_fn((output_size, input_size + 1))

        # TODO: set the bias terms to zero
        self.w[:, 0] = 0

        # TODO: Initialize matrix to store gradient with respect to weights
        self.dw = np.zeros((output_size, input_size + 1))

        # TODO: Initialize any additional values you may need to store for the
        #  backward pass here
        self.x = np.array([])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input to linear layer with shape (input_size,)
                  where input_size *does not include* the folded bias.
                  In other words, the input does not contain the bias column
                  and you will need to add it in yourself in this method.
                  Since we train on 1 example at a time, batch_size should be 1
                  at training.
        :return: output z of linear layer with shape (output_size,)

        HINT: You may want to cache some of the values you compute in this
        function. Inspect your expressions for backprop to see which values
        should be cached.
        """
        # TODO: perform forward pass and save any values you may need for
        #  the backward pass
        self.x = np.concatenate([np.ones(1), x])
        return self.w @ self.x

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output z
            of linear
        :return: dx, partial derivative of loss with respect to input x
            of linear

        Note that this function should set self.dw
            (gradient of loss with respect to weights)
            but not directly modify self.w; NN.step() is responsible for
            updating the weights.

        HINT: You may want to use some of the values you previously cached in
        your forward() method.
        """
        self.dw = np.outer(dz, self.x)
        return self.w[:, 1:].T @ dz

    def step(self) -> None:
        """
        Apply SGD update to weights using self.dw, which should have been
        set in NN.backward().
        """
        # TODO: implement
        self.w -= self.lr * self.dw


class NN:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weight_init_fn: INIT_FN_TYPE,
        learning_rate: float,
    ):
        """
        Initalize neural network (NN) class. Note that this class is composed
        of the layer objects (Linear, Sigmoid) defined above.

        :param input_size: number of units in input to network
        :param hidden_size: number of units in the hidden layer of the network
        :param output_size: number of units in output of the network - this
                            should be equal to the number of classes
        :param weight_init_fn: function that creates and initializes weight
                               matrices for layer. This function takes in a
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        # TODO: initialize modules (see section 9.1.2 of the writeup)
        #  Hint: use the classes you've implemented above!
        self.linear1 = Linear(input_size, hidden_size, weight_init_fn, learning_rate)
        self.sigmoid = Sigmoid()
        self.linear2 = Linear(hidden_size, output_size, weight_init_fn, learning_rate)
        self.loss_fn = SoftMaxCrossEntropy()

    def forward(self, x: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Neural network forward computation.
        Follow the pseudocode!
        :param x: input data point *without the bias folded in*
        :param y: prediction with shape (num_classes,)
        :return:
            y_hat: output prediction with shape (num_classes,). This should be
                a valid probability distribution over the classes.
            loss: the cross_entropy loss for a given example
        """
        # TODO: call forward pass for each layer
        a = self.linear1.forward(x)
        z = self.sigmoid.forward(a)
        b = self.linear2.forward(z)
        y_hat, loss = self.loss_fn.forward(b, y)
        return y_hat, loss

    def backward(self, y: int, y_hat: np.ndarray) -> None:
        """
        Neural network backward computation.
        Follow the pseudocode!
        :param y: label (a number or an array containing a single element)
        :param y_hat: prediction with shape (num_classes,)
        """
        # TODO: call backward pass for each layer
        grad_b = self.loss_fn.backward(y, y_hat)
        grad_z = self.linear2.backward(grad_b)
        grad_a = self.sigmoid.backward(grad_z)
        _ = self.linear1.backward(grad_a)

    def step(self):
        """
        Apply SGD update to weights.
        """
        # TODO: call step for each relevant layer
        self.linear2.step()
        self.linear1.step()

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute nn's average (cross entropy) loss over the dataset (X, y)
        :param X: Input dataset of shape (num_points, input_size)
        :param y: Input labels of shape (num_points,)
        :return: Mean cross entropy loss
        """
        # TODO: compute loss over the entire dataset
        #  Hint: reuse your forward function
        num_points = y.shape[0]
        losses = 0.0
        for i in range(num_points):
            _, loss = self.forward(X[i], y[i])
            losses += loss
        return losses / num_points

    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_epochs: int,
    ) -> Tuple[List[float], List[float]]:
        """
        Train the network using SGD for some epochs.
        :param X_tr: train data
        :param y_tr: train label
        :param X_test: train data
        :param y_test: train label
        :param n_epochs: number of epochs to train for
        :return:
            train_losses: Training losses *after* each training epoch
            test_losses: Test losses *after* each training epoch
        """
        # TODO: train network
        train_losses = [0.0] * n_epochs
        test_losses = [0.0] * n_epochs
        for e in range(n_epochs):
            X_shuffled, y_shuffled = shuffle(X_tr, y_tr, e)
            for x, y in zip(X_shuffled, y_shuffled):
                y_hat, _ = self.forward(x, y)
                self.backward(y, y_hat)
                self.step()
            train_losses[e] = self.compute_loss(X_shuffled, y_shuffled)
            test_losses[e] = self.compute_loss(X_test, y_test)
        return train_losses, test_losses

    def test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the label and error rate.
        :param X: input data
        :param y: label
        :return:
            labels: predicted labels
            error_rate: prediction error rate
        """
        # TODO: make predictions and compute error
        labels = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            y_hat, _ = self.forward(x, y[i])
            labels[i] = np.argmax(y_hat)
        error_rate = np.mean(labels != y)
        return labels, error_rate


if __name__ == "__main__":
    args = parser.parse_args()
    # Note: You can access arguments like learning rate with args.learning_rate
    # Generally, you can access each argument using the name that was passed
    # into parser.add_argument() above (see lines 24-44).

    # Define our labels
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]

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

    nn = NN(
        input_size=X_tr.shape[-1],
        hidden_size=n_hid,
        output_size=len(labels),
        weight_init_fn=zero_init if init_flag == 2 else random_init,
        learning_rate=lr,
    )

    # train model
    # (this line of code is already written for you)
    train_losses, test_losses = nn.train(X_tr, y_tr, X_test, y_test, n_epochs)

    # test model and get predicted labels and errors
    # (this line of code is written for you)
    train_labels, train_error_rate = nn.test(X_tr, y_tr)
    test_labels, test_error_rate = nn.test(X_test, y_test)

    # Write predicted label and error into file
    # Note that this assumes train_losses and test_losses are lists of floats
    # containing the per-epoch loss values.
    with open(out_tr, "w") as f:
        for label in train_labels:
            f.write(str(label) + "\n")
    with open(out_te, "w") as f:
        for label in test_labels:
            f.write(str(label) + "\n")
    with open(out_metrics, "w") as f:
        for i in range(len(train_losses)):
            cur_epoch = i + 1
            cur_tr_loss = train_losses[i]
            cur_te_loss = test_losses[i]
            f.write("epoch={} crossentropy(train): {}\n".format(cur_epoch, cur_tr_loss))
            f.write(
                "epoch={} crossentropy(validation): {}\n".format(cur_epoch, cur_te_loss)
            )
        f.write("error(train): {}\n".format(train_error_rate))
        f.write("error(validation): {}\n".format(test_error_rate))
