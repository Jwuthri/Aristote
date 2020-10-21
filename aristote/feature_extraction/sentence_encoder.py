import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub

from aristote.utils import predict_format


class SentenceEncoder(object):

    def __init__(self, name="pretrained_embedding", embedding_path=None):
        self.name = name
        self.embedding_path = embedding_path
        self.model = self.get_model()

    def get_model(self):
        hub_path = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        bert_module = self.embedding_path if self.embedding_path else hub_path
        embedding_model = hub.load(bert_module)

        return hub.KerasLayer(embedding_model, input_shape=[], dtype=tf.string, trainable=False, name=self.name)

    @predict_format
    def predict_one(self, text):
        return self.model(text)

    @predict_format
    def predict_batch(self, text):
        raise self.model(text)
