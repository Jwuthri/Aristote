import os

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'

import pandas as pd

from aristote.settings import DATASET_PATH
from aristote.tensorflow_helper.trainer_helper import TensorflowTrainer


if __name__ == '__main__':
    dataset_path = os.path.join(DATASET_PATH, "sentiment.csv")
    architecture = [('LCNN', 512), ("GLOBAL_AVERAGE_POOL", 0), ("DROPOUT", 0.1), ('DENSE', 256)]
    dataset = pd.read_csv(dataset_path)
    x_col, y_col, label_type, epochs, name = "feature", "multi", "multi-label", 5, "sentiment"

    trainer = TensorflowTrainer(label_type, name, architecture, epochs=epochs)
    trainer.train(dataset, x_col, y_col)
