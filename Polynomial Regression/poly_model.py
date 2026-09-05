import numpy as np

class PolynomialRegression:
    def __init__(self,degree = 2,learning_rate = 0.01,epochs = 1000):
        self.W = None
        self.b = 0.0
        self.losses = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mean = 0
        self.std = 0
        self.degree = degree

    def predict(self,X):
        X_poly = self.PolynomialFeatures(X)
        X_scaled = (X_poly - self.mean) / (self.std +1e-8)
        return X_scaled @ self.W + self.b

    def cost_func(self,Y,Y_hat):
        n = Y.shape[0]
        self.losses.append(np.sum((Y_hat - Y)**2)/(2*n))
    def initialize_params(self,X):
        self.W = np.random.randn(X.shape[1],1)
        print(self.W)
    def PolynomialFeatures(self,X):
        n_col = X.shape[1]
        columns = [X]
        for i in range(n_col):
            x = X[:,i]
            poly = x.copy().reshape(-1,1)

            for j in range(2,self.degree+1):
                columns.append(poly**j)
        return np.column_stack(columns)

    def fit(self,X,Y):
        n = Y.shape[0]
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        X_poly = self.PolynomialFeatures(X)

        self.mean = np.mean(X_poly,axis=0)
        self.std = np.std(X_poly,axis=0)

        X_scaled = (X_poly - self.mean) / (self.std + 1e-8)
        self.initialize_params(X_scaled)
        for i in range(self.epochs):
            Y_hat = X_scaled @ self.W + self.b

            dw = np.dot(X_scaled.T,(Y_hat - Y)) / n
            db = np.sum(Y - Y_hat) / n

            self.W = self.W - (self.learning_rate * dw)
            self.b = self.b - (self.learning_rate * db)

            self.cost_func(Y,Y_hat)

            if i % 50 == 0 or i == self.epochs - 1:
                print(f"Epoch {i}: Loss {self.losses[-1]}")



