"""
In this script, the Server code using FedAVG is provided.
Remember to set the hyperparameters for the training.
"""
import flwr as fl
import tensorflow as tf
import json
import os
import sys
import flwr as fl
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

min_available_clients = 21
n_rounds = 200
early_stopping_rounds = 15


# create an empty json file where to store all the results during each round
results_path = r"model_weights\results.json"
results_dict = {
    "round": [],
    "n_clients": [],
    "total_samples": [],
    "train_loss": [],
    "val_loss": []
}
with open(results_path, "w") as json_file:
    json.dump(results_dict, json_file)

    
# aggregate all the clients results using simply the mean (ex: mean of all the train loss)
def aggregate_losses(reports: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    train_losses = [report["train_loss"] for report in reports if "train_loss" in report]
    val_losses = [report["val_loss"] for report in reports if "val_loss" in report]
    aggregated_metrics = {}
    if train_losses:
        aggregated_metrics["mean_train_loss"] = np.mean(train_losses)
    if val_losses:
        aggregated_metrics["mean_val_loss"] = np.mean(val_losses)
    return aggregated_metrics if aggregated_metrics else None

# the CustomFedAvg using FedAVG
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, eta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = eta  # Reptile step size
        self.theta_weights = None
        self.best_train_loss = float('inf')
        self.consecutive_no_improvement = 0
        self.early_stopping_rounds = 15

    # aggregate the results during the eval phase
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[Any, Dict[str, Any], int]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Dict[str, Any], int]]:
        eval_results  = [result[1] for result in results]


        # aggregate losses and metrics
        total_loss = np.mean([res.loss for res in eval_results])
        total_evaluation_examples = sum(res.num_examples for res in eval_results)
        eval_metrics = [res.metrics for res in eval_results]

        # aggregate with the custom metrics defined in the previous function (e.g., train_loss, val_loss)
        aggregated_metrics = aggregate_losses(eval_metrics)

        # create a result dictionary
        result_dict = {
            "round": rnd,
            "train_loss": aggregated_metrics.get('mean_train_loss', 'N/A'),
            "val_loss": aggregated_metrics.get('mean_val_loss', 'N/A'),
            "n_clients": len(eval_results),
            "total_samples": total_evaluation_examples
        }

        print(f"Client Results: {result_dict['n_clients']}, "
              f"Total Samples: {result_dict['total_samples']}, "
              f"Round {result_dict['round']}: Aggregated Metrics - "
              f"Mean Train Loss: {result_dict['train_loss']}, "
              f"Mean Val Loss: {result_dict['val_loss']}")
        with open(results_path, "r") as json_file:
            results_dict = json.load(json_file)
        results_dict["round"].append(result_dict["round"])
        results_dict["n_clients"].append(result_dict["n_clients"])
        results_dict["total_samples"].append(result_dict["total_samples"])
        results_dict["train_loss"].append(result_dict["train_loss"])
        results_dict["val_loss"].append(result_dict["val_loss"])
        
        # early stopping phase
        with open(results_path, "w") as json_file:
            json.dump(results_dict, json_file)

        if result_dict['train_loss'] != 'N/A':
            if result_dict['train_loss'] >= self.best_train_loss:
                self.consecutive_no_improvement += 1
                if self.consecutive_no_improvement >= self.early_stopping_rounds: sys.exit(0)
            else:
                self.consecutive_no_improvement = 0
                self.best_train_loss = result_dict['train_loss']

        return total_loss, aggregated_metrics



# create an instance of the custom strategy
strategy = CustomFedAvg(
    min_available_clients=min_available_clients,
    #fraction_fit=0.4
)

# start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=n_rounds),
    strategy=strategy,
)