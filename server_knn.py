import flwr as fl
import utils_knn
from sklearn.metrics import log_loss
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict

def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: KNeighborsClassifier):
    """Return an evaluation function for server-side evaluation."""
    _, (X_test, y_test) = utils_knn.iris_data()

    def evaluate(parameters: fl.common.Weights):
        utils_knn.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = KNeighborsClassifier(n_neighbors=3)
    utils_knn.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    #print(strategy)
    fl.server.start_server("localhost:8080", strategy=strategy, config={"num_rounds": 2})
