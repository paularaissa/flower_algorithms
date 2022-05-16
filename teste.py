from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

irisData = load_iris()

#print(irisData)

# Create feature and target arrays
X = irisData.data
y = irisData.target

#print(X)
#print(len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(type(X_train))

teste = np.array_split(X_train, 2)

x_train, y_train = X[:200], y[:200]

print(teste)