import numpy as np

class LinearRegression:
    def __init__(self,learning_rate = 0.01,epochs = 1000):
        self.W = None
        self.b = 0.0
        self.losses = []
        self.learning_rate = learning_rate
        self.epochs = epochs



    def LinearModel(self,X):
        return X @ self.W + self.b

    def cost_func(self,Y,Y_hat):
        n = Y.shape[0]
        self.losses.append(np.sum((Y_hat - Y)**2)/(2*n))
    def initialize_params(self,X):
        self.W = np.random.randn(X.shape[1],1)

    def updating_weights(self,X,Y):
        n = Y.shape[0]
        Y_hat = self.LinearModel(X)

        return np.dot(X.T,(Y_hat - Y)) / n

    def update_bias(self,X,Y):
        n = Y.shape[0]
        Y_hat = self.LinearModel(X)

        return np.sum(Y_hat - Y) / n




    def fit(self,X,Y):
        n = Y.shape[0]
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.initialize_params(X)
        for i in range(self.epochs):
            dw = self.updating_weights(X,Y)
            db =  self.update_bias(X,Y)

            self.W = self.W - (self.learning_rate * dw)
            self.b = self.b - (self.learning_rate * db)

            Y_hat  = self.LinearModel(X)
            self.cost_func(Y,Y_hat)

            if i % 50 == 0 or i == self.epochs - 1:
                print(f"Epoch {i}: Loss {self.losses[-1]}")



