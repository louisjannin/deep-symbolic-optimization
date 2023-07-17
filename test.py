import numpy as np
from dso import DeepSymbolicRegressor
import matplotlib.pyplot as plt

# Generate some data
np.random.seed(0)
X = np.random.random((50, 2))
y = np.sin(X[:,0]) + X[:,1] ** 2
# y = np.atleast_2d(y).T
# data = np.concatenate((X, y), axis=1)
# np.savetxt('data/dataset.csv', data, delimiter=',')

# Create the model
model = DeepSymbolicRegressor('config.json') # Alternatively, you can pass in your own config JSON path
# model = DeepSymbolicRegressor() # Alternatively, you can pass in your own config JSON path

# Fit the model
# model.fit(X, y) # Should solve in ~10 seconds
model.train() # Should solve in ~10 seconds

# Make predictions
# model.predict(2 * X)
