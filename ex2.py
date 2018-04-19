import numpy as np
from matplotlib import pyplot as plt
from random import shuffle


def normal_distribution(points, mu):
    """
    Normal distribution function.
    :param points: set of points
    :param mu: mu
    :return:
    """
    return 1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-(points - mu) ** 2 / (2 * 1 ** 2))


def softmax(W, b, x):
    """
    Softmax function.
    :param W: weights matrix
    :param b: bias vector
    :param x: example input
    :return: matrix of probabilities
    """
    numerator = np.exp(np.dot(W, x) + b)
    denominator = 0

    for j in range(W.shape[0]):
        denominator += np.exp(W[j] * x + b[j])

    return numerator / denominator


def predict(W, b, x):
    """
    Predicts to which class the given examples belongs.
    :param W: Weights matrix
    :param b: bias vector
    :param x: example
    :return: predicted class
    """
    return np.argmax(softmax(W, b, x)) + 1


def update(W, x, b, y, eta):
    """
    Updates the weights and the bias vector according to the update rule.
    :param W: weights
    :param x: input example
    :param b: bias vector
    :param y: input label
    :param eta: learning rate
    :return: None
    """
    length = W.shape[0]

    for i in range(length):
        denominator = 0

        for j in range(length):
            denominator += np.exp(W[j] * x + b[j])

        numerator = np.exp(W[i] * x + b[i])

        softmax_val = numerator / denominator

        if i == y:
            W[i] -= -eta * x + eta * softmax_val * x
            b[i] -= eta * softmax_val - eta
        else:
            W[i] -= eta * softmax_val * x
            b[i] -= eta * softmax_val


def calc_loss(y, y_hat):
    """
    Calculates the error.
    :param y: true value
    :param y_hat: predicted value
    :return:None
    """
    if y != y_hat:
        try:
            calc_loss.counter += 1
        except AttributeError:
            calc_loss.counter = 1
        print calc_loss.counter


def train_logistic_regression(training_examples, num_of_labels, dimension):
    """
    Performs multiclass logistic regression on the given training examples.
    :param training_examples: set of training examples
    :param num_of_labels: number of labels
    :param dimension: dimension
    :return: trained weights and bias
    """
    # Learning rate.
    eta = 0.1

    # Number of epochs.
    epochs = 100

    # Weights matrix.
    W = np.ones(shape=(num_of_labels, dimension))

    # Bias vector.
    b = np.ones(shape=(num_of_labels, 1))

    for e in range(epochs):
        # Shuffle examples.
        shuffle(training_examples)

        for (x, y) in training_examples:
            # Predict.
            y_hat = predict(W, b, x)

            # Used for debugging.
            # calc_loss(y, y_hat)

            # Perform update.
            update(W, x, b, y - 1, eta)

    return W, b


def approximate_to_one(points, W, b):
    """
    Perform approximation to p( y = 1 | x)
    :param points: set of points
    :param W: weights matrix
    :param b: bias vector
    :return: approximation vector
    """
    return [softmax(W, b, x)[0] for x in points]


if __name__ == "__main__":
    # For label a=1, taking 100 samples.
    x1 = np.random.normal(2 * 1, 1, 100)

    # For label a=2, taking 100 samples.
    x2 = np.random.normal(2 * 2, 1, 100)

    # For label a=3, taking 100 samples.
    x3 = np.random.normal(2 * 3, 1, 100)

    # A training set with the above samples and their labels.
    training_examples = [(x, 1) for x in x1]
    training_examples += [(x, 2) for x in x2]
    training_examples += [(x, 3) for x in x3]

    # Create 100 points in the range[0,10]
    # TODO should it be 0,11 instead of 10?
    points = np.linspace(0, 10, 150)

    # Calculate the true distribution for p(y = 1 | x).
    true_distribution = normal_distribution(points, 2) / (
        normal_distribution(points, 2) + normal_distribution(points, 4) + normal_distribution(points, 6))

    # Train the logistic regression.
    W, b = train_logistic_regression(training_examples, 3, 1)

    # Approximate the distribution for p(y = 1 | x) according to the learned regression.
    approximated_distribution = approximate_to_one(points, W, b)

    # Plot the graphs.
    plt.plot(points, true_distribution, 'b', label='true distribution')
    plt.plot(points, approximated_distribution, 'r--', label='learned distribution')

    plt.ylabel('Probability')
    plt.xlabel('Points')
    plt.legend()

    plt.show()
