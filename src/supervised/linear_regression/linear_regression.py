import numpy as np

class Model():

    def __init__(self, epochs=None, lr=None, normal_equation=False):
        self.epochs = epochs
        self.lr = lr
        self.normal_equation = normal_equation
        
    def __initialize_weights(self, nb_features):
        """
            Initialize weights from uniform distribution.
        """
        bound = 1 / np.sqrt(nb_features)
        self.W = np.random.uniform(low=-bound, high=bound, size=(nb_features,))
        
    def fit(self, X, y):
        # Insert constant bias at first column.
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.__initialize_weights(nb_features=X.shape[1])

        if self.normal_equation:
            X_t = np.transpose(X)
            inv = np.linalg.inv
            self.W = inv(X_t @ X) @ (X_t @ y)
        else:
            m = X.shape[1]
            for epoch in range(self.epochs):
                y_pred = X @ self.W
                cost = (1/(2*m)) * sum(y_pred - y)**2
                self.training_errors.append(cost) 
                self.grad_W = (1/m) * (y_pred - y).dot(X)
                self.W -= self.lr * self.grad_W

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X @ self.W
