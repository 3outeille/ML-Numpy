import numpy as np

class Sigmoid():

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

class Model():

    def __init__(self, epochs=None, lr=None):
        self.epochs = epochs
        self.lr = lr
        self.sigmoid = Sigmoid()
    
    def __initialize_weights(self, nb_features):
        """
                Initialize weights from uniform distribution.
        """  
        bound = 1 / np.sqrt(nb_features)
        self.W = {'val': np.random.uniform(low=-bound, high=bound, size=(nb_features,1)),
                  'grad': np.zeros((nb_features,1))}

    def fit(self, X, y):
        # Insert constant bias at first column.
        X = np.insert(X, 0, 1, axis=1)
        m = X.shape[1]
        self.__initialize_weights(nb_features=m)
        self.training_errors = []

        for epoch in range(self.epochs):
            # Forward pass.
            y_pred = self.sigmoid(X @ self.W['val'])
            # Compute Cost.
            cost = (-1/m) * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
            self.training_errors.append(cost) 
            # Backward pass.
            self.W['grad'] = (1/m) * X.T @ (y_pred-y)
            self.W['val'] -= self.lr * self.W['grad']
            
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return self.sigmoid(X @ self.W['val'])

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

# Dataset visualization
X = np.array([
            [1, 4],
            [2, 4],
            [3, 4],
            [4, 4],
            [5, 4],
            [6, 10],
            [7, 10],
            [8, 10],
            [9, 10],
            [10, 10]
        ])
y = np.array([1,1,1,1,1,0,0,0,0,0])
y = y[:, np.newaxis]

plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], c=y[:, 0], s=20, cmap='jet')
# plt.show()

# Model initialization
model = Model(epochs=1000, lr=0.01)
model.fit(X, y)
y_pred = model.predict(X)
