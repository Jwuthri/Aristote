import os
import json
import pickle
import shutil
import datetime

import pandas as pd

import tensorflow as tf

from aristote.settings import MODEL_PATH


class TensorflowLoaderSaver(object):
    """Module to save or load a model saved."""

    def __init__(self, name, model_load, **kwargs):
        self.name = name
        self.model_load = model_load
        self.model_info = dict()
        self.base_path = kwargs.get('base_path', MODEL_PATH)
        self.date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.paths = self.define_paths()

    @staticmethod
    def use_gpu(use_gpu=False):
        if use_gpu:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            tf.config.set_visible_devices([], 'GPU')

    def define_paths(self):
        if self.model_load:
            base_dir = os.path.join(self.base_path, self.name)
        else:
            base_dir = os.path.join(self.base_path, f"{self.date}_{self.name}")

        return {
            "path": base_dir,
            "model_path": os.path.join(base_dir, 'model.pkl'),
            "weights_path":  os.path.join(base_dir, 'weights'),
            "metrics_path": os.path.join(base_dir, "metrics.json"),
            "model_plot_path":  os.path.join(base_dir, "model.jpg"),
            "model_info_path": os.path.join(base_dir, "model.json"),
            "optimizer_score_path": os.path.join(base_dir, "optimizer_scores.csv"),
            "checkpoint_path": os.path.join(base_dir, "checkpoint"),
            "tokenizer_path": os.path.join(base_dir, "tokenizer.pkl"),
            "tensorboard_path": os.path.join(base_dir, "tensorboard"),
            "label_encoder_path": os.path.join(base_dir, "label_encoder.pkl"),
            "classes_thresholds_path": os.path.join(base_dir, "classes_thresholds.json"),
        }

    def export_tf_model_plot(self, model):
        tf.keras.utils.plot_model(model, to_file=self.paths['model_plot_path'], show_shapes=True)

    def export_weights(self, model):
        model.save_weights(self.paths['weights_path'])
        print(f"Model has been exported here => {self.paths['weights_path']}")

    def load_weights(self, model):
        latest = tf.train.latest_checkpoint(self.paths['path'])
        model.load_weights(latest)

        return model

    def export_label_encoder(self, label_encoder):
        pickle.dump(label_encoder, open(self.paths['label_encoder_path'], "wb"))
        print(f"Label Encoder has been exported here => {self.paths['label_encoder_path']}")

    def load_label_encoder(self):
        encoder = pickle.load(open(self.paths['label_encoder_path'], "rb"))

        return encoder

    def export_info(self, info):
        with open(self.paths['model_info_path'], 'w') as outfile:
            json.dump(info, outfile)
            print(f"Model information have been exported here => {self.paths['model_info_path']}")

    def load_info(self):
        with open(self.paths['model_info_path'], "rb") as json_file:
            info = json.load(json_file)

        return info

    def export_metrics(self, metrics):
        with open(self.paths['metrics_path'], 'w') as outfile:
            json.dump(metrics, outfile)
            print(f"Model metrics have been exported here => {self.paths['metrics_path']}")

    def load_metrics(self):
        with open(self.paths['metrics_path'], "rb") as json_file:
            metrics = json.load(json_file)

        return metrics

    def export_tokenizer(self, tokenizer):
        pickle.dump(tokenizer, open(self.paths['tokenizer_path'], "wb"))
        print(f"Tokenizer has been exported here => {self.paths['tokenizer_path']}")

    def load_tokenizer(self):
        tokenizer = pickle.load(open(self.paths['tokenizer_path'], "rb"))

        return tokenizer

    def export_model(self, model):
        pickle.dump(model, open(self.paths['model_path'], "wb"))
        print(f"Model has been exported here => {self.paths['model_path']}")

    def load_model(self):
        model = pickle.load(open(self.paths['model_path'], "rb"))

        return model

    def export_optimizer_score(self, scores):
        scores.to_csv(self.paths['optimizer_score_path'], index=False)
        print(f"Optimizer scores have been exported here => {self.paths['optimizer_score_path']}")

    def load_optimizer_score(self):
        return pd.read_csv(self.paths['optimizer_score_path'])

    def export_thresholds(self, thresholds):
        with open(self.paths['classes_thresholds_path'], 'w') as outfile:
            json.dump(thresholds, outfile)
            print(f"Classes thresholds have been exported here => {self.paths['classes_thresholds_path']}")

    def load_thresholds(self):
        with open(self.paths['classes_thresholds_path'], "rb") as json_file:
            thresholds = json.load(json_file)

        return thresholds

    def zip_model(self):
        zip_path = shutil.make_archive(
            self.paths['path'], "zip", os.path.dirname(self.paths['path']), os.path.basename(self.paths['path'])
        )
        print(f"Zip has been generated here => {zip_path}")

        return zip_path
