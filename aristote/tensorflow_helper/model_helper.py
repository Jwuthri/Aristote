from aristote.tensorflow_helper.layers_helper import get_input_layer, text_to_layer, get_output_layer
from aristote.tensorflow_helper.saver_helper import TensorflowLoaderSaver
from aristote.tensorflow_helper.losses_helper import *
from aristote.tensorflow_helper.metrics_helper import f1_score


class TensorflowModel(TensorflowLoaderSaver):

    def __init__(self, architecture, label_type, name, **kwargs):
        self.architecture = architecture
        self.label_type = label_type
        self.label_encoder_classes_number = kwargs.get('label_encoder_classes_number', 1)
        self.input_size = kwargs.get('input_size', 128)
        self.embedding_size = kwargs.get('embedding_size', 256)
        self.vocab_size = kwargs.get('vocab_size', 200000)
        self.pretrained_embedding = kwargs.get('pretrained_embedding', True)
        self.model = tf.keras.Sequential()
        model_load = kwargs.pop('model_load', False)
        super().__init__(name, model_load, **kwargs)
        exception_message = f"label_type should be in ['binary-class', 'multi-label', 'multi-class'] not {label_type}"
        assert label_type in ['binary-class', 'multi-label', 'multi-class'], exception_message

    def build_model(self):
        input_layer, layer = get_input_layer(self.pretrained_embedding, self.embedding_size, self.input_size, self.vocab_size)
        self.model_info['input_size'] = self.input_size
        self.model_info['vocab_size'] = self.vocab_size
        self.model_info['embedding_size'] = self.embedding_size
        self.model_info['pretrained_embedding'] = self.pretrained_embedding
        self.model_info['architecture'] = self.architecture
        self.model_info['label_type'] = self.label_type
        self.model_info['label_encoder_classes_number'] = self.label_encoder_classes_number
        block = None
        for block, unit in self.architecture:
            layer = text_to_layer(block, unit)(layer)
        use_time_distrib = True if block == "TIME_DISTRIB_DENSE" else False
        output_layer = get_output_layer(
            self.label_type, units=self.label_encoder_classes_number, use_time_distrib=use_time_distrib)(layer)
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    def compile_model(self):
        if self.label_type == "binary-class":
            self.model.compile(
                optimizer="rmsprop",
                loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
                metrics=["binary_accuracy", "Recall", "Precision"])
        elif self.label_type == "multi-label":
            self.model.compile(
                optimizer="adam",
                loss=f1_loss,
                metrics=[f1_score, "categorical_accuracy", "top_k_categorical_accuracy"])
        else:
            self.model.compile(
                optimizer="adam",
                loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                metrics=["categorical_accuracy", "top_k_categorical_accuracy"])
