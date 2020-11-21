import numpy as np

from aristote.tensorflow_helper.saver_helper import TensorflowLoaderSaver
from aristote.tensorflow_helper.model_helper import TensorflowModel
from aristote.preprocessing.normalization import TextNormalization
from aristote.utils import predict_format


class TensorflowPredictor(TensorflowModel, TensorflowLoaderSaver):
    """Module to predict saved model."""

    def __init__(self, name, model_load, cleaning_func=None, **kwargs):
        self.name = name
        self.cleaning_func = cleaning_func
        self.normalizer = TextNormalization()
        TensorflowLoaderSaver.__init__(self, name, model_load, **kwargs)
        self.info = self.load_info()
        self.info['model_load'] = model_load
        self.label_encoder = self.load_label_encoder()
        self.classes_thresholds = self.load_thresholds()
        TensorflowModel.__init__(self, name=name, **self.info)
        self.build_model()
        self.load_weights(self.model)
        breakpoint()

    def clean_text(self, text):
        text = self.normalizer.replace_char_rep(text=text)
        text = self.normalizer.replace_words_rep(text=text)
        if self.cleaning_func:
            text = self.cleaning_func(text)
        text = self.normalizer.remove_multiple_spaces(text=text)
        text = text.strip()

        return text

    @predict_format
    def predict_multi_label(self, text, thresholds=None):
        cleaned = np.asarray([self.clean_text(x) for x in text])
        thresholds = thresholds if thresholds else self.classes_thresholds
        predictions = [dict(zip(self.label_encoder.classes_.tolist(), x)) for x in self.model.predict(cleaned).tolist()]
        filtered_predictions = [
            [label for label, proba in prediction.items() if proba >= thresholds[label]] for prediction in predictions
        ]

        return filtered_predictions

    @predict_format
    def predict_multi_class(self, text):
        cleaned = np.asarray([self.clean_text(x) for x in text])
        predictions = self.model.predict(cleaned).tolist()
        predictions = [self.label_encoder.classes_[np.argmax(prediction)] for prediction in predictions]

        return predictions

    @predict_format
    def predict(self, text):
        cleaned = np.asarray([self.clean_text(x) for x in text])
        predictions = [dict(zip(self.label_encoder.classes_.tolist(), x)) for x in self.model.predict(cleaned).tolist()]
        filtered_predictions = [
            [(label, proba) for label, proba in prediction.items() if proba >= self.classes_thresholds[label]] for prediction in predictions
        ]

        return filtered_predictions
