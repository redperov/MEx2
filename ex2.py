import numpy as np
from matplotlib import pyplot as plt
from random import shuffle

def normal_distribution(points, a):
    return 1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-(points - a) ** 2 / (2 * 1 ** 2))


def softmax(W, b, x):
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


# def softmax(W, x, b):
#
#     # TODO if doesn't work, try the above function
#     numerator = np.exp(np.dot(W, x) + b)
#     denominator = 0
#
#     for j in range(W.shape[1]):
#         denominator += np.exp(W[j] * x + b[j])
#
#     return numerator / denominator


def update(W, x, b, y, eta):
    length = W.shape[0]

    for i in range(length):

        denominator = 0
        for j in range(length):
            denominator += np.exp(W[j] * x + b[j])

        numerator = np.exp(W[i] * x + b[i])

        if i == y:
            W[i] = -eta * x + eta * numerator / denominator * x
            b[i] = eta * numerator / denominator - eta
        else:
            W[i] = eta * numerator / denominator * x
            b[i] = eta * numerator / denominator


def calc_loss(W, b, training_examples):

    loss_sum = 0
    for (x, y) in training_examples:
        loss_sum += -np.log(softmax(W, b, x)[y - 1])

    return loss_sum


def train_logistic(training_examples, num_of_labels, dimension):
    # Learning rate.
    eta = 0.1

    # Number of epochs.
    epochs = 10

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

            # print calc_loss(W, b, training_examples)

            # Check if need to update.
            if y_hat != y:
                update(W, x, b, y - 1, eta)

    return W, b


def approximate_to_one(points, W, b):
    return [softmax(W, b, x)[0] for x in points]


if __name__ == "__main__":
    # For label a=1, taking 100 samples.
    x1 = np.random.normal(2 * 1, 1, 100)

    # For label a=2, taking 100 samples.
    x2 = np.random.normal(2 * 2, 1, 100)

    # For label a=3, taking 100 samples.
    x3 = np.random.normal(2 * 3, 1, 100)

    # X_train = np.array([x1, x2, x3])
    # Y_train = np.array([1, 2, 3])

    training_examples = [(x, 1) for x in x1]
    training_examples += [(x, 2) for x in x2]
    training_examples += [(x, 3) for x in x3]
    # training_examples = np.concatenate((xy1, xy2, xy3))

    # Create 100 points in the range[0,10]
    # TODO should it be 0,11 instead of 10?
    points = np.linspace(0, 10, 100)

    # Calculate the true distribution for p(y = 1 | x).
    true_distribution = normal_distribution(points, 2) / (
        normal_distribution(points, 2) + normal_distribution(points, 4) + normal_distribution(points, 6))

    # Train the logistic regression.
    W, b = train_logistic(training_examples, 3, 1)

    # Approximate the distribution for p(y = 1 | x) according to the learned regression.
    approximated_distribution = approximate_to_one(points, W, b)

    # Plot the graphs.
    plt.plot(points, true_distribution, 'b', label='true distribution')
    plt.plot(points, approximated_distribution, 'r--', label='learned distribution')

    plt.ylabel('Probability')
    plt.xlabel('Points')
    plt.legend()

    plt.show()
