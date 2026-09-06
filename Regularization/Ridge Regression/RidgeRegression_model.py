import numpy as np
class RidgeRegression:
    def __init__(self,lamda = 1 ,learning_rate = 0.3,epochs = 400):
        self.W = None
        self.losses = []
        self.b = 0.0
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.epochs = epochs

    def initialize_params(self,X):
        self.W = np.random.randn(X.shape[1],1)


    def predict(self,X):
        return X@self.W + self.b

    def cost_func(self,X,Y,Y_hat):
        n = Y.shape[0]
        first_term = np.sum((Y_hat - Y)**2) / (2 * n)

        L2_penality = ((self.lamda /(2*n) ) *np.sum( self.W**2) )
        self.losses.append(first_term  + L2_penality)

    def fit(self,X,Y):
        # Ensure Y is a column vector (n, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.initialize_params(X)
        for i in range(self.epochs):
            Y_hat = self.predict(X)
            n = Y.shape[0]
            dw = (np.dot(X.T,Y_hat-Y) / n) + (self.lamda/n * self.W)
            db = np.sum(Y_hat - Y) / n

            self.W = self.W - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            self.cost_func(X, Y, Y_hat)


            if i % 50 == 0 or i == self.epochs - 1:
                print(f"Epoch {i}: Loss {self.losses[-1]}")



