import re

from tqdm import tqdm

import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from aristote.utils import timer
from aristote.tensorflow_helper.dataset_helper import TensorflowDataset


class GenerationDataset(TensorflowDataset):

    def __init__(self, architecture, label_type, name, **kwargs):
        self.tokenizer = Tokenizer(filters='', num_words=self.num_words, oov_token='[UNK]')
        self.max_labels = kwargs.get("max_labels", len(self.tokenizer.index_word))
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

    @timer
    def generate_generation_dataset(self, data, x_column, y_column=None):
        data = data[data[x_column].notnull()]
        cleaned_data = self.clean_x(data[x_column], remove_emoji=True, start_end_token=True)
        self.set_tokenizer(" ".join(cleaned_data))
        labels = self.predictable_words()
        x_train, x_val = train_test_split(cleaned_data, test_size=self.test_size, random_state=42)
        train_dataset = self.generation_dataset(x_train, labels, dataset_name="train")
        val_dataset = self.generation_dataset(x_val, labels, dataset_name="validation", is_training=False)

        return train_dataset, val_dataset
