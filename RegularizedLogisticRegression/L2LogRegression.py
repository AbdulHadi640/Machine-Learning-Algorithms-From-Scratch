import numpy as np
class L2LogisticRegression:
    def __init__(self,learning_rate = 0.1, epochs = 500, lamda = 1):
        self.W = 0
        self.b  = 0.0
        self.losses = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lamda = lamda
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def predict(self,X):

        linear = X @ self.W + self.b
        Y_hat = self.sigmoid(linear)

        return Y_hat

    def cost_function(self,Y,Y_hat):
        n = Y.shape[0]
        eps = 1e-15
        Y_hat = np.clip(Y_hat, eps, 1 - eps)
        bce = (-1 / n) * np.sum( Y * np.log(Y_hat) +(1 - Y) * np.log(1 - Y_hat))
        L2_penality = ((self.lamda /(2*n) ) *np.sum( self.W**2) )

        loss = bce+L2_penality
        self.losses.append(loss)

    def initialize_parameters(self,X_shape):
        self.W = np.random.randn(X_shape[1], 1)
        self.b = np.random.randn(1,1)

    def gradient(self,X,Y,Y_hat):
        n = Y.shape[0]
        dw = (np.dot(X.T, Y_hat - Y) / n) + (self.lamda / n * self.W)
        db = np.sum(Y_hat - Y) / n

        return dw,db
    def fit(self,X,Y):

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.initialize_parameters(X.shape)

        for i in range(self.epochs):
            Y_hat = self.predict(X)
            self.cost_function(Y, Y_hat)

            dw,db = self.gradient(X,Y,Y_hat)

            self.W = self.W - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            if i % 50 == 0 or i == self.epochs - 1:
                print(f"Loss after epoch {i} is {self.losses[-1]}")

