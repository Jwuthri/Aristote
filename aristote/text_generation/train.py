import os

import pandas as pd

from aristote.settings import DATASET_PATH
from aristote.tensorflow_helper.trainer_helper import TensorflowTrainer


if __name__ == '__main__':
    dataset_path = os.path.join(DATASET_PATH, "phrase_prediction_experiment_datasets_fr_dataset_sep_23.csv")
    architecture = [('LCNN', 512), ("GLOBAL_AVERAGE_POOL", 0), ("DROPOUT", 0.1), ('DENSE', 256)]
    dataset = pd.read_csv(dataset_path, nrows=5000)
    x_col, y_col, epochs, name = "processed", None, 2, "generation"

    trainer = TensorflowTrainer(
        "multi-class", name, architecture, epochs=epochs, model_type="generation", pretrained_embedding=False)
    trainer.train(dataset, x_col, y_col)
