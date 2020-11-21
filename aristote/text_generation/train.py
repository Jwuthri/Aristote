import os

import pandas as pd

from aristote.settings import DATASET_PATH
from aristote.tensorflow_helper.trainer_helper import TensorflowTrainer


if __name__ == '__main__':
    dataset_path = os.path.join(DATASET_PATH, "text_generation_french.csv")
    architecture = [('LCNN', 512), ("GLOBAL_AVERAGE_POOL", 0), ("DROPOUT", 0.1), ('DENSE', 256)]
    dataset = pd.read_csv(dataset_path)
    x_col, y_col, label_type, epochs, name = "processed", None, "multi-class", 5, "phrase"

    trainer = TensorflowTrainer(label_type, name, architecture, epochs=epochs)
    trainer.train(dataset, x_col, y_col)
