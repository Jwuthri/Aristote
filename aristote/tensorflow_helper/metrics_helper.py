import numpy as np

import tensorflow as tf


@tf.function
def f1_score(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)"""
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)

    return macro_f1


@tf.function
def hamming_score(y_true, y_pred):
    """Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case"""
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)

    return np.mean(acc_list)


def get_metrics(history, metric="f1_score"):
    """Extract the loss and metric from the model"""
    train_loss = [float(x) for x in history.history["loss"]]
    val_loss = [float(x) for x in history.history["val_loss"]]
    train_metric = [float(x) for x in history.history[metric]]
    val_metric = [float(x) for x in history.history[f"val_{metric}"]]

    return {"train_loss": train_loss, "val_loss": val_loss, "train_metric": train_metric, "val_metric": val_metric}


class BinaryTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name="binary_true_positives", **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives


class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name="f1_score", **kwargs):
        """initializes attributes of the class"""

        # call the parent class init
        super(F1Score, self).__init__(name=name, **kwargs)
        self.tp = tf.Variable(0, dtype="int32")
        self.fp = tf.Variable(0, dtype="int32")
        self.tn = tf.Variable(0, dtype="int32")
        self.fn = tf.Variable(0, dtype="int32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates statistics for the metric

        Args:
            y_true: target values from the test data
            y_pred: predicted values by the model
        """
        # Calulcate confusion matrix.
        conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=2)

        # Update values of true positives, true negatives, false positives and false negatives from confusion matrix.
        self.tn.assign_add(conf_matrix[0][0])
        self.tp.assign_add(conf_matrix[1][1])
        self.fp.assign_add(conf_matrix[0][1])
        self.fn.assign_add(conf_matrix[1][0])

    def result(self):
        """Computes and returns the metric value tensor."""

        # Calculate precision
        if (self.tp + self.fp == 0):
            precision = 1.0
        else:
            precision = self.tp / (self.tp + self.fp)

        # Calculate recall
        if (self.tp + self.fn == 0):
            recall = 1.0
        else:
            recall = self.tp / (self.tp + self.fn)

        # Return F1 Score
        f1_score = 2 * ((precision * recall) / (precision + recall))

        return f1_score

    def reset_states(self):
        """Resets all of the metric state variables."""

        # The state of the metric will be reset at the start of each epoch.
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)
