import warnings
import flwr as fl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils_knn

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = utils_knn.iris_data()

    partition_id = np.random.choice(10)
    (X_train, y_train) = utils_knn.partition(X_train, y_train, 10)[partition_id]
    model = KNeighborsClassifier(n_neighbors=3)

utils_knn.set_initial_params(model)

class IrisClient(fl.client.NumPyClient):
    def get_parameters(self): # type: ignore
        return utils_knn.get_model_parameters(model)

    def fit(self, parameters, config):# type: ignore
        print("fit")
        utils_knn.set_model_params(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        print(f"Training finished for round {config['rnd']}")
        return utils_knn.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        utils_knn.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}

fl.client.start_numpy_client("localhost:8080", client=IrisClient())