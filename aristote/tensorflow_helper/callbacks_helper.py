import matplotlib.pyplot as plt
import datetime
import imageio
import io

from IPython.display import Image as IPyImage
from PIL import Image
import numpy as np

from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
import tensorflow as tf


def frobenius_callback(model):
    """https://www.tensorflow.org/addons/tutorials/optimizers_conditionalgradient"""

    def normalize(m):
        """This function is to calculate the frobenius norm of the matrix of all layer's weight."""
        total_reduce_sum = 0
        for i in range(len(m)):
            total_reduce_sum = total_reduce_sum + tf.math.reduce_sum(m[i] ** 2)
        norm = total_reduce_sum ** 0.5

        return norm

    normalized = []
    weights = model.trainable_weights

    return LambdaCallback(on_epoch_end=lambda batch, logs: normalized.append(normalize(weights).numpy()))


def checkpoint_callback(ckpt_path):
    """Create basic callbacks for tensorflow."""
    return ModelCheckpoint(filepath=ckpt_path, verbose=1, period=1, save_weights_only=True)


def tensorboard_callback(tsboard_path=None):
    """Create basic callbacks for tensorflow."""
    return TensorBoard(log_dir=tsboard_path, histogram_freq=1)


class VisCallback(tf.keras.callbacks.Callback):

    def __init__(self, inputs, ground_truth, display_freq=10, n_samples=10):
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.images = []
        self.display_freq = display_freq
        self.n_samples = n_samples
        self.GIF_PATH = './animation.gif'

    @staticmethod
    def display_digits(inputs, outputs, ground_truth, epoch, n=10):
        plt.clf()
        plt.yticks([])
        plt.grid(None)
        inputs = np.reshape(inputs, [n, 28, 28])
        inputs = np.swapaxes(inputs, 0, 1)
        inputs = np.reshape(inputs, [28, 28 * n])
        plt.imshow(inputs)
        plt.xticks([28 * x + 14 for x in range(n)], outputs)
        plt.title(epoch)
        for i, t in enumerate(plt.gca().xaxis.get_ticklabels()):
            if outputs[i] == ground_truth[i]:
                t.set_color('green')
            else:
                t.set_color('red')
        plt.grid(None)

    def on_epoch_end(self, epoch, logs=None):
        indexes = np.random.choice(len(self.inputs), size=self.n_samples)
        x_test, y_test = self.inputs[indexes], self.ground_truth[indexes]
        predictions = np.argmax(self.model.predict(x_test), axis=1)
        self.display_digits(x_test, predictions, y_test, epoch, n=self.display_freq)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.images.append(np.array(image))
        if epoch % self.display_freq == 0:
            plt.show()

    def on_train_end(self, logs=None):
        imageio.mimsave(self.GIF_PATH, self.images, fps=1)

    def create_gif(self):
        IPyImage(self.GIF_PATH, format='png', width=15 * 60, height=3 * 60)


class MyCustomCallback(tf.keras.callbacks.Callback):

    def on_train_batch_begin(self, batch, logs=None):
        print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    def on_train_batch_end(self, batch, logs=None):
        print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass
