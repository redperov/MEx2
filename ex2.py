import numpy as np
from matplotlib import pyplot as plt
from random import shuffle


def normal_distribution(points, a):
    return 1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-(points - a) ** 2 / (2 * 1 ** 2))


def softmax(W, x, b, i):

    numerator = np.exp(W[i] * x + b[i])
    denominator = 0

    for j in range(W.shape[1]):
        denominator += np.exp(W[j] * x + b[j])

    return numerator / denominator

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

        if i == y:
            W[i] = -eta * x + (eta * np.exp(W[i] * x + b[i])) / denominator
            b[i] = eta * (np.exp(W[i] * x + b[i])) / denominator - eta
        else:
            W[i] = (eta * np.exp(W[i] * x + b[i])) / denominator
            b[i] = eta * (np.exp(W[i] * x + b[i])) / denominator


def train_logistic(X_train, Y_train):
    # Learning rate.
    eta = 0.1

    # Number of epochs.
    epochs = 10

    num_of_labels = X_train.shape[0]
    dimension = 1

    # Weights matrix.
    W = np.ones(shape=(num_of_labels, dimension))

    # Bias vector.
    b = np.ones(shape=(num_of_labels, 1))

    for e in range(epochs):
        # X_train, Y_train = shuffle(X_train, Y_train, random_state=1)
        #i = 0
        for x, y in zip(X_train, Y_train):
            # Predict.
            y_hat = softmax(W, x, b, y)
            # TODO remove if not needed i += 1
            # Check if need to update.
            if y_hat != y:
                update(W, x, b, y, eta)

    return W, b


def approximate(points, y, W, b):
    return [softmax(W, x, b, y) for x in points]


if __name__ == "__main__":

    # For label a=1, taking 100 samples.
    x1 = np.random.normal(2 * 1, 1, 100)

    # For label a=2, taking 100 samples.
    x2 = np.random.normal(2 * 2, 1, 100)

    # For label a=3, taking 100 samples.
    x3 = np.random.normal(2 * 3, 1, 100)

    # Create 100 points in the range[0,10]
    # TODO should it be 0,11 instead of 10?
    points = np.linspace(0, 10, 100)

    X_train = np.array([x1, x2, x3])
    Y_train = np.array([1, 2, 3])

    # Calculate the true distribution for p(y = 1 | x).
    true_distribution = normal_distribution(points, 2) / (
        normal_distribution(points, 2) + normal_distribution(points, 4) + normal_distribution(points, 6))

    # Train the logistic regression.
    W, b = train_logistic(X_train, Y_train)

    # Approximate the distribution for p(y = 1 | x) according to the learned regression.
    approximated_distribution = approximate(points, 1, W, b)

    # Plot the graphs.
    plt.plot(points, true_distribution, 'b', label='true distribution')
    plt.plot(points, approximated_distribution, 'r', label='learned distribution')

    plt.ylabel('Probability')
    plt.xlabel('Points')
    plt.legend()

    plt.show()

