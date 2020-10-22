import os

import pandas as pd

from bayes_opt import BayesianOptimization
from sklearn.metrics import precision_recall_fscore_support

from aristote.settings import DATASET_PATH
from aristote.tensorflow_helper.dataset_helper import TensorflowDataset
from aristote.tensorflow_helper.predictor_helper import TensorflowPredictor


class BayesianOptimizer(TensorflowPredictor):
    """Optimizer for thresholds."""

    def __init__(self, name, model_load, dataset_path, x, y, **kwargs):
        self.dataset = pd.read_csv(dataset_path, nrows=100)
        self.x = x
        self.y = y
        super().__init__(name, model_load, **kwargs)
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def black_box_function(self, average="weighted", metric='f1', **thresholds):
        """Test on new thresholds to see the improvements."""
        y_pred = self.predict_multi_label(text=self.dataset[self.x].values, thresholds=thresholds)
        y_true = TensorflowDataset.clean_y(self.dataset[self.y].values)
        encoded_y_true = [[1 if label in row else 0 for label in self.label_encoder.classes_] for row in y_true]
        encoded_y_pred = [[1 if label in row else 0 for label in self.label_encoder.classes_] for row in y_pred]
        score = precision_recall_fscore_support(encoded_y_true, encoded_y_pred, average=average)
        score = dict(zip(['precisiom', 'recall', 'f1', 'support'], score))

        return score[metric]

    def optimize(self, init_points=10, n_iter=20, acq="ei", kappa=0.1, alpha=1e-3, n_restarts_optimizer=5):
        """Bayesian Grid search for thresholds optimization"""
        assert acq in ["ucb", "poi", "ei"], "Please select one of [ucb, poi, ei]"
        pbounds = {k: [0.1, 1.0] for k in self.classes_thresholds.keys()}
        optimizer = BayesianOptimization(f=self.black_box_function, pbounds=pbounds, random_state=42)
        optimizer.maximize(
            init_points=init_points, n_iter=n_iter, acq=acq, kappa=kappa,
            alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)

        return optimizer

    def bayesian_optimizer(self):
        """Find optimal thresholds."""
        opt = self.optimize()
        results = pd.DataFrame(opt.res)
        params_score = pd.concat([results["params"].apply(pd.Series), results.target], axis=1)
        print(params_score)

        return params_score


if __name__ == '__main__':
    data_path = os.path.join(DATASET_PATH, "sentiment.csv")
    model_path = "2020_10_15_16_04_09_sentiment"
    bo = BayesianOptimizer(name=model_path, model_load=True, dataset_path=data_path, x="feature", y="multi")
    bo.bayesian_optimizer()
