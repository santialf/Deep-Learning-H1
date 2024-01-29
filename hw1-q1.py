#!/usr/bin/env python

# Deep Learning Homework 1
# Santiago and Alina
import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_hat = self.predict(x_i)
        
        if y_hat != y_i:
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i
        
        # Q1.1a
        #raise NotImplementedError


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # softmax
        prob = np.exp(self.W.dot(x_i)[:, None])
        y_hat = prob / np.sum(prob)

        # true label for correct value
        y_true = np.zeros((np.size(self.W, 0), 1))
        y_true[y_i] = 1
        
        # compute the gradient
        grad = (y_true - y_hat) * x_i[None, :]
        
        # SGD update.
        self.W += learning_rate * grad
        
        # Q1.1b
        #raise NotImplementedError


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers):
        self.layers = layers
        self.W1 = np.random.normal(0.1, 0.1, (hidden_size, n_features))
        self.b1 = np.zeros((hidden_size))
        self.W2 = np.random.normal(0.1, 0.1, (n_classes, hidden_size))
        self.b2 = np.zeros((n_classes))
        self.W = [self.W1, self.W2]
        self.B = [self.b1, self.b2]
        # Initialize an MLP with a single hidden layer.
        #raise NotImplementedError

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        hiddens = []
        
        # forward pass
        for i in range(self.layers + 1):
            h = X if i == 0 else hiddens[i-1]
            z = h.dot(self.W[i].T) + self.B[i]
            # Apply activation function to hidden layer
            if i < self.layers:
                hiddens.append(np.maximum(0.0, z))
        
        # z is the output matrix (10000x10)
        # y_hat is the predicted output array (size 10000)
        y_hat = np.zeros(len(z))
        
        # pick the value with highest sum
        for i in range(len(y_hat)):
            y_hat[i] = np.argmax(z[i])
        
        return y_hat
        #raise NotImplementedError

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        # every example
        for j in range(0, X.shape[0]):
            
            # only sample 1 training example, SGD
            x_i = X[j]
            y_i = y[j]
            hiddens = []
            
            # forward pass
            for i in range(self.layers + 1):
                h = x_i if i == 0 else hiddens[i-1]
                z = self.W[i].dot(h) + self.B[i]
                # Apply activation function to hidden layer
                if i < self.layers:
                    hiddens.append(np.maximum(0.0, z))
            
            # softmax
            probs = np.exp(z- np.max(z)) / np.sum(np.exp(z- np.max(z)))
            
            # true label for correct value
            y_true = np.zeros_like(probs)
            y_true[y_i] = 1
            
            # gradient
            grad_z = probs - y_true
            
            grad_weights = []
            grad_biases = []

            # backpropagation
            for i in range(self.layers, -1, -1):
                # Gradient of hidden parameters.
                h = x_i if i == 0 else hiddens[i-1]
                grad_weights.append(grad_z[:, None].dot(h[:, None].T))
                grad_biases.append(grad_z)

                # Gradient of hidden layer below.
                grad_h = self.W[i].T.dot(grad_z)

                # Gradient of hidden layer below before activation.
                grad_z = grad_h * (h>0)  

            grad_weights.reverse()
            grad_biases.reverse()

            # update parameters
            for i in range(self.layers+1):
                self.W[i] -= learning_rate*grad_weights[i]
                self.B[i] -= learning_rate*grad_biases[i]

        return
        #raise NotImplementedError


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
