from typing import Tuple, Union, List
import numpy as np
#from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import openml

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: KNeighborsClassifier) -> LogRegParams:
    """Returns the paramters of a sklearn KNeighborsClassifier model."""

    params = model.n_neighbors

    #if model.n_neighbors:
    #    params = (model.n_neighbors, model.weights)

    return params


def set_model_params(
    model: KNeighborsClassifier, params: LogRegParams
) -> KNeighborsClassifier:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.n_neighbors = params

    #if model.fit_intercept:
    #    model.intercept_ = params[1]
    return model


def set_initial_params(model: KNeighborsClassifier):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 3  #Flower has 5 classes
    n_features = 4  # Number of features in dataset
    model.classes_ = np.array([i for i in range(3)])

    #model.coef_ = np.zeros((n_classes, n_features))
    model.n_neighbors = 3

    #if model.fit_intercept:
    #    model.intercept_ = np.zeros((n_classes,))
    #model.n_jobs = -1

def iris_data() -> Dataset:
    """Loads the MNIST dataset using OpenML.

    OpenML dataset link: https://www.openml.org/d/554
    """
    # mnist_openml = openml.datasets.get_dataset(554)
    # Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    # X = Xy[:, :-1]  # the last column contains labels
    # y = Xy[:, -1]
    # First 60000 samples consist of the train set
    #x_train, y_train = X[:60000], y[:60000]
    #x_test, y_test = X[60000:], y[60000:]
    irisData = load_iris()
    # Create feature and target arrays
    X = irisData.data
    y = irisData.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return (x_train, y_train), (x_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
