from abc import ABC

from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
import tensorflow as tf


class EncoderNetwork(tf.keras.Model, ABC):
    """Encoder module."""

    def __init__(self, input_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size, output_dim=embedding_dims)
        self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True)


class DecoderNetwork(tf.keras.Model, ABC):
    """Decoder module."""

    def __init__(self, output_vocab_size, embedding_dims, rnn_units, dense_units, batch_size,  attention_method):
        super().__init__()
        self.attention_method = attention_method
        self.dense_units = dense_units
        self.batch_size = batch_size
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size, output_dim=embedding_dims)
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnn = tf.keras.layers.LSTMCell(rnn_units)
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.attention_mechanism = self.build_attention_mechanism(dense_units, None, self.batch_size * [Tx])
        self.rnn_cell = self.build_rnn_cell()
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.dense_layer)

    def build_attention_mechanism(self, units, memory, memory_sequence_length):
        if self.attention_method == "BahdanauAttention":
            return tfa.seq2seq.BahdanauAttention(units, memory=memory, memory_sequence_length=memory_sequence_length)
        else:
            return tfa.seq2seq.LuongAttention(units, memory=memory, memory_sequence_length=memory_sequence_length)

    def build_rnn_cell(self):
        rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnn, self.attention_mechanism, attention_layer_size=self.dense_units
        )

        return rnn_cell

    def build_decoder_initial_state(self, batch_size, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

        return decoder_initial_state
