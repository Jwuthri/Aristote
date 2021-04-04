import re
import ast

from tqdm import tqdm
import plotly.express as px

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from aristote.utils import timer
from aristote.preprocessing.normalization import TextNormalization
from aristote.tensorflow_helper.model_helper import TensorflowModel

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class TensorflowDataset(TensorflowModel):
    """Module to generate dataset."""

    def __init__(self, architecture, label_type, name, **kwargs):
        self.test_size = kwargs.get('test_size', 0.15)
        self.num_words = kwargs.get("num_words", 35000)
        self.batch_size = kwargs.get('batch_size', 256)
        self.buffer_size = kwargs.get('buffer_size', 256)
        self.input_shape = kwargs.get('input_shape', 128)
        self.max_labels = kwargs.get('max_labels', 2000)
        self.plot_distribution = kwargs.get('plot_distribution', False)
        self.vocab_size = 0
        self.number_labels = 0
        self.classes_thresholds = dict()
        self.label_encoder_classes = dict()
        self.label_encoder_classes_number = 0
        self.label_encoder = self.init_label_encoder(label_type)
        self.tokenizer = Tokenizer(filters='', num_words=self.num_words, oov_token='[UNK]')
        self.splitter = "|".join([
            "!", "@", "#", "$", "%", "^", "&", "\\(", "\\)", "_", "-", ",", "<", "\\.", ">", "\\?", "`", "~", ":",
            ";", "\\+", "=", "[", "]", "{", "}", "\n{2,}", "\\s", "\n"
        ])
        self.start, self.end, self.separator = "[START]", "[END]", "[SEP]"
        super().__init__(architecture, label_type, name, **kwargs)

    def split_text(self, text):
        text = text.lower()

        return re.sub(r'(' + self.splitter + ')', r' \1 ', text)

    def predictable_words(self):
        self.max_labels = len(self.tokenizer.index_word) if not self.max_labels else self.max_labels
        self.number_labels = min(self.max_labels, len(self.tokenizer.index_word))
        words = list(range(1, self.number_labels))

        return words

    def set_tokenizer(self, text):
        self.tokenizer.fit_on_texts([text])
        self.vocab_size = len(self.tokenizer.word_index) + 1

    @staticmethod
    def init_label_encoder(label_type):
        if label_type == "multi-label":
            return MultiLabelBinarizer()
        else:
            return LabelBinarizer()

    def fit_encoder(self, y):
        self.label_encoder.fit(y)
        self.label_encoder_classes = dict(enumerate(self.label_encoder.classes_))
        self.classes_thresholds = {cls: 0.5 for cls in self.label_encoder.classes_}
        self.label_encoder_classes_number = len(self.label_encoder.classes_)

    @staticmethod
    def clean_x(x, remove_emoji=False, start_end_token=False):
        pbar = tqdm(total=len(x), desc="Cleaning x")
        cleaned_texts = []
        tn = TextNormalization()
        for text in x:
            if remove_emoji:
                text = tn.text_demojis(text=text)
                text = tn.text_demoticons(text=text)
            text = tn.replace_char_rep(text=text)
            text = tn.replace_words_rep(text=text)
            text = tn.remove_multiple_spaces(text=text)
            text = text.strip()
            if start_end_token:
                start, end, separator = "[START]", "[END]", "[SEP]"
                text = f"{start} {text} {end}"
            cleaned_texts.append(text)
            pbar.update(1)
        pbar.close()

        return cleaned_texts

    @staticmethod
    def clean_y(y):
        pbar = tqdm(total=len(y), desc="Cleaning y")
        cleaned_labels = []
        for label in y:
            try:
                label = ast.literal_eval(label)
            except ValueError:
                label = [label]
            cleaned_labels.append(label)
            pbar.update(1)
        pbar.close()

        return cleaned_labels

    def to_tensorflow_dataset(self, x, y, dataset_name="train", is_training=True):
        pbar = tqdm(total=1, desc=f"Generating dataset {dataset_name}")
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if is_training:
            dataset = dataset.cache()
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        pbar.update(1)
        pbar.close()

        return dataset

    def texts_to_sequences(self, data, labels, dataset_name):
        texts_to_sequences = list()
        pbar = tqdm(total=len(data), desc=f"Creating the dataset {dataset_name}")
        for text in data:
            encoded_text = self.tokenizer.texts_to_sequences([text])[0]
            for idx in range(1, len(encoded_text) - 1):
                if encoded_text[idx:idx + 1][0] in labels:
                    texts_to_sequences.append(encoded_text[:idx + 1])
            pbar.update(1)
        pbar.close()

        return texts_to_sequences

    def generation_dataset(self, data, labels, dataset_name="train", is_training=True):
        texts_to_sequences = self.texts_to_sequences(data, labels, dataset_name)
        padded_sequences = np.array(pad_sequences(texts_to_sequences, maxlen=self.input_shape + 1, padding='pre'))
        x, y = padded_sequences[:, :-1], padded_sequences[:, -1:]
        dataset = self.to_tensorflow_dataset(x, y, dataset_name, is_training)

        return dataset

    @staticmethod
    def plot_classes_distribution(data, column):
        fig = px.histogram(data, x=column)
        fig.show()

    @timer
    def generate_classification_dataset(self, data, x_column, y_column):
        data = data[data[x_column].notnull()]
        data = data[data[y_column].notnull()]
        if self.plot_distribution:
            self.plot_classes_distribution(data, y_column)
        # x = self.clean_x(data[x_column])
        x = data[x_column]
        y = self.clean_y(data[y_column]) if self.label_type == "multi-label" else data[y_column]
        self.fit_encoder(y)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.test_size, random_state=42)
        y_train_encoded = self.label_encoder.transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        train_dataset = self.to_tensorflow_dataset(x_train, y_train_encoded, dataset_name="train")
        val_dataset = self.to_tensorflow_dataset(x_val, y_val_encoded, dataset_name="validation", is_training=False)

        return train_dataset, val_dataset

    @timer
    def generate_generation_dataset(self, data, x_column, y_column=None):
        data = data[data[x_column].notnull()]
        cleaned_data = self.clean_x(data[x_column], remove_emoji=True, start_end_token=True)
        self.set_tokenizer(" ".join(cleaned_data))
        labels = self.predictable_words()
        x_train, x_val = train_test_split(cleaned_data, test_size=self.test_size, random_state=42)
        self.label_encoder_classes_number = len(labels) + 1
        train_dataset = self.generation_dataset(x_train, labels, dataset_name="train")
        val_dataset = self.generation_dataset(x_val, labels, dataset_name="validation", is_training=False)

        return train_dataset, val_dataset
