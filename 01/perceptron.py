# Gangadhara, Karthik
# 1001-677-851
# 2019-09-23
# Assignment-01-01

import numpy as np
import itertools

class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self._initialize_weights()
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = []
        self.weights = np.random.randn(self.number_of_classes, self.input_dimensions + 1)

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = []
        self.weights = np.zeros((self.number_of_classes, self.input_dimensions + 1))

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]
        """
        X_bias_row = np.ones((1,X.shape[1]))
        X = np.concatenate((X_bias_row,X))
        Product = np.dot(self.weights,X)
        result = np.zeros((Product.shape))
        for i in range(Product.shape[0]):
            for j in range(Product.shape[1]):
                if Product[i][j] >= 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0

        return result    



    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)


    def train(self, X, Y, num_epochs=10, alpha=0.001):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        X_bias_row = np.ones((1,X.shape[1]))
        X = np.concatenate((X_bias_row,X))
        X_transpose = X.transpose()
        Y_transpose = Y.transpose()

        for epochs in range(num_epochs):
            for y_row, x_row in zip(Y_transpose,X_transpose):
                y_row_tranpose = y_row.reshape(y_row.shape+(1,))
                x_row_tranpose = x_row.reshape(x_row.shape+(1,))

                a = np.matmul(self.weights,x_row_tranpose)
                for row in range(a.shape[0]):
                    for col in range(a.shape[1]):
                        if a[row][col] >= 0:
                            a[row][col] = 1
                        else:
                            a[row][col] = 0
                e = y_row_tranpose - a
                self.weights = self.weights + alpha * np.matmul(e,x_row_tranpose.transpose())

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the output is not hte same as the desired output, Y,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :return percent_error
        """
        X_bias_row = np.ones((1,X.shape[1]))
        X = np.concatenate((X_bias_row,X))
        X_transpose = X.transpose()
        Y_transpose = Y.transpose()
        number_of_errors = 0 
        number_of_samples = Y.shape[1]
        for y_row, x_row in zip(Y_transpose,X_transpose):
            y_row_tranpose = y_row.reshape(y_row.shape+(1,))
            x_row_tranpose = x_row.reshape(x_row.shape+(1,))

            a = np.matmul(self.weights,x_row_tranpose)
            for row in range(a.shape[0]):
                for col in range(a.shape[1]):
                    if a[row][col] >= 0:
                        a[row][col] = 1
                    else:
                        a[row][col] = 0
            e = y_row_tranpose - a
            zero_count = np.count_nonzero(e)
            if zero_count >= 1: 
                number_of_errors += 1

        percent_error = number_of_errors/ number_of_samples
        return percent_error

if __name__ == "__main__":
    """
    This main program is a sample of how to run your program.
    You may modify this main program as you desire.
    """

    input_dimensions = 2
    number_of_classes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    model.initialize_all_weights_to_zeros()
    print("****** Model weights ******\n",model.weights)
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.weights)