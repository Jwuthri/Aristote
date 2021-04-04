import re
import string
import operator

import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
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
        self.max_labels = self.info['max_labels']
        self.label_encoder = self.load_label_encoder()
        self.classes_thresholds = self.load_thresholds()
        self.tokenizer = self.load_tokenizer()
        TensorflowModel.__init__(self, name=name, **self.info)
        self.build_model()
        self.load_weights(self.model)
        self.splitter = "|".join([
            "!", "@", "#", "$", "%", "^", "&", "\\(", "\\)", "_", "-", ",", "<", "\\.", ">", "\\?", "`", "~", ":",
            ";", "\\+", "=", "[", "]", "{", "}", "\n{2,}", "\\s", "\n"
        ])
        self.possible_words = list(self.tokenizer.word_index.keys())[:self.max_labels]
        self.no_completion = (',', '!', '?', ':', ';', '.', '\n', '\n\n', '', ' ')
        self.end_sentence = ('!', '?', '.', '\n', '\n\n')
        self.no_space_needed = (" ", "\n", "\n\n")
        self.splitter = " "

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
            [(label, proba) for label, proba in prediction.items() if proba >= 0.1]  # self.classes_thresholds[label]]
            for prediction in predictions
        ]

        return filtered_predictions

    def split_text(self, text):
        text = text.lower()

        return re.sub(r'(' + self.splitter + ')', r' \1 ', text)

    def to_sequence(self, text):
        tokenized_text = self.tokenizer.texts_to_sequences([text])[0]
        padded_tokenized_text = pad_sequences([tokenized_text], maxlen=self.input_size, padding='pre')

        return padded_tokenized_text

    def split_text_last_word(self, text):
        return text.split(self.splitter)[-1], " ".join(text.split(self.splitter)[:-1])

    def get_prediction_startwith(self, predictions, word, threshold=0.3):
        tokens_ids = np.where(predictions[0] >= threshold)[0]
        scores = [predictions[0][idx] for idx in tokens_ids]
        token_score = list(zip(tokens_ids, scores))
        token_score_possible = dict(
            [(self.tokenizer.index_word.get(token), score) for (token, score) in token_score if score >= threshold])
        sorted_token = sorted(token_score_possible.items(), key=operator.itemgetter(1), reverse=True)
        for token, score in sorted_token:
            if token.startswith(word):
                return token, score

        return "", 0.0

    def word_completion(self, text, threshold=0.3):
        score, token = 0.0, ""
        if text[-1] == self.splitter:
            return text, [score], [token]
        split_text = self.split_text(text)
        last_word, begin_text = self.split_text_last_word(split_text)
        cleaned_text = self.clean_text(begin_text)
        if last_word not in self.no_completion:
            padded_tokenized_text = self.to_sequence(cleaned_text)
            predictions = self.model.predict(padded_tokenized_text)
            token, score = self.get_prediction_startwith(predictions,last_word, threshold)
            text += token[len(last_word):]

        return text, [score], [token[len(last_word):]]

    def next_word(self, text):
        split_text = self.split_text(text)
        last_word = split_text.split(self.splitter)[-1]
        cleaned_text = self.clean_text(split_text)
        tokenized_text = self.tokenizer.texts_to_sequences([cleaned_text])[0]
        padded_tokenized_text = pad_sequences([tokenized_text], maxlen=self.input_size, padding='pre')
        predictions = self.model.predict(padded_tokenized_text)
        prediction = np.argmax(predictions[0])
        token = self.tokenizer.index_word.get(prediction)
        score = predictions[0][prediction]
        if (token not in string.punctuation) and (token not in self.no_space_needed) and (text[-1] != self.splitter):
            if last_word in self.end_sentence:
                token = token.capitalize()
            token = " " + token

        return score, token

    def next_words(self, text, threshold=0.4, max_predictions=5):
        scores, tokens, score, number_predicted_words, continue_prediction = [], [], 1.0, 0, True
        while continue_prediction:
            score_token, token = self.next_word(text)
            score *= score_token
            if score >= threshold:
                scores.append(score)
                tokens.append(token)
                number_predicted_words += 1
                text += token
            else:
                continue_prediction = False
            if number_predicted_words == max_predictions:
                continue_prediction = False

        return text, scores, tokens

    def generate_text(self, x, completion_threshold=0.2, prediction_threshold=0.3, max_predictions=5):
        text = x
        text, completion_score, completion_token = self.word_completion(text, threshold=completion_threshold)
        text, next_word_scores, next_word_tokens = self.next_words(
            text, threshold=prediction_threshold, max_predictions=max_predictions)
        predicted_text = text[len(x):]
        breakpoint()

        return predicted_text
