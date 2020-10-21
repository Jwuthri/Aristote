import operator

import numpy as np

from aristote.tensorflow_helper.saver_helper import TensorflowLoaderSaver
from aristote.tensorflow_helper.model_helper import TensorflowModel
from aristote.utils import predict_format


class TensorflowPredictor(TensorflowModel, TensorflowLoaderSaver):
    """Module to predict saved model."""

    def __init__(self, name, model_load, **kwargs):
        self.name = name
        TensorflowLoaderSaver.__init__(self, name, model_load, **kwargs)
        self.info = self.load_info()
        self.info['model_load'] = model_load
        self.label_encoder = self.load_label_encoder()
        self.classes_thresholds = self.load_thresholds()
        TensorflowModel.__init__(self, name=name, **self.info)
        self.build_model()
        self.load_weights(self.model)

    @predict_format
    def predict_multi_label(self, text, thresholds=None):
        thresholds = thresholds if thresholds else self.classes_thresholds
        predictions = [dict(zip(self.label_encoder.classes_.tolist(), x)) for x in self.model.predict(text).tolist()]
        filtered_predictions = [
            [label for label, proba in prediction.items() if proba >= thresholds[label]] for prediction in predictions
        ]

        return filtered_predictions

    @predict_format
    def predict_multi_class(self, text):
        predictions = self.model.predict(text).tolist()
        predictions = [self.label_encoder.classes_[np.argmax(prediction)] for prediction in predictions]
        # filtered_predictions = sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)[0]

        return predictions

    @predict_format
    def predict(self, text):
        predictions = [dict(zip(self.label_encoder.classes_.tolist(), x)) for x in self.model.predict(text).tolist()]
        filtered_predictions = [
            [(label, proba) for label, proba in prediction.items() if proba >= self.classes_thresholds[label]] for prediction in predictions
        ]

        return filtered_predictions
