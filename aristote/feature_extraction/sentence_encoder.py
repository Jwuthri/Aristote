import bert

import numpy as np

import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub

from aristote.utils import predict_format


class MUSE(object):

    def __init__(self, trainable=False):
        self.trainable = trainable
        self.model = self.get_model()

    def get_model(self):
        embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

        return hub.KerasLayer(embedding_model, input_shape=[], dtype=tf.string, trainable=self.trainable)

    @predict_format
    def predict_one(self, text):
        return self.model(text)

    @predict_format
    def predict_batch(self, text):
        raise self.model(text)


class LaBSE(object):

    def __init__(self, trainable=False, padding="right", max_seq_length=256):
        assert padding in ["right", "left"]
        self.trainable = trainable
        self.padding = padding
        self.max_seq_length = max_seq_length
        self.labse_model, self.labse_layer = self.get_model()
        self.vocab_file = self.labse_layer.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.labse_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = bert.bert_tokenization.FullTokenizer(self.vocab_file, self.do_lower_case)

    def get_model(self):
        labse_layer = hub.KerasLayer("https://tfhub.dev/google/LaBSE/1", trainable=self.trainable)
        input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="segment_ids")

        pooled_output, _ = labse_layer([input_word_ids, input_mask, segment_ids])
        pooled_output = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)
        model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=pooled_output)

        return model, labse_layer

    def create_input(self, input_strings):
        input_ids_all, input_mask_all, segment_ids_all = [], [], []
        for input_string in input_strings:
            input_tokens = ["[CLS]"] + self.tokenizer.tokenize(input_string) + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            sequence_length = min(len(input_ids), self.max_seq_length)

            if self.padding == "right":
                if len(input_ids) >= self.max_seq_length:
                    input_ids = input_ids[:self.max_seq_length]
                else:
                    input_ids = input_ids + [0] * (self.max_seq_length - len(input_ids))
            else:
                if len(input_ids) >= self.max_seq_length:
                    input_ids = input_ids[-self.max_seq_length:]
                else:
                    input_ids = [0] * (self.max_seq_length - len(input_ids)) + input_ids

            input_mask = [1] * sequence_length + [0] * (self.max_seq_length - sequence_length)
            input_ids_all.append(input_ids)
            input_mask_all.append(input_mask)
            segment_ids_all.append([0] * self.max_seq_length)

        return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)

    @predict_format
    def predict_one(self, text):
        input_ids, input_mask, segment_ids = self.create_input(text)

        return self.labse_model([input_ids, input_mask, segment_ids])
