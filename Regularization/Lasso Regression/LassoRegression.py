import numpy as np

class LassoRegression:
    def __init__(self,lamda = 2, learning_rate = 0.01, epochs = 10000):
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = 0
        self.losses = []

    def LinearModel(self,X):
        return X @ self.W + self.b

    def initialize_weights(self,n_weights):
        self.W = np.random.randn(n_weights,1)

    def compute_cost(self,Y, Y_hat):
        n = Y.shape[0]
        return (1/(2*n))*np.sum((Y_hat - Y)**2) + ((self.lamda / n)*np.sum(np.abs(self.W)))



    def update_weights(self,X,Y):
        n = X.shape[0]
        Y_hat = self.LinearModel(X)

        return (np.dot(X.T,Y_hat- Y)) / n + (self.lamda/n) * np.sign(self.W)

    def update_bias(self,X,Y):
        n = X.shape[0]
        Y_hat = self.LinearModel(X)

        return np.sum(Y_hat -  Y) / n

    def fit(self,X,Y):
        self.initialize_weights(X.shape[1])


        for epoch in range(self.epochs):
            dw = self.update_weights(X,Y)
            db = self.update_bias(X,Y)

            self.W = self.W - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            Y_hat = self.LinearModel(X)

            self.losses.append(self.compute_cost(Y,Y_hat))

            if epoch % 50 == 0 or epoch == (self.epochs - 1):
                print(f"Cost after epoch {epoch}: {self.losses[-1]}")






