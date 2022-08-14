import tensorflow as tf
import numpy as np

class MnistDataset():
    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size

    def load(self):
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        all_digits = np.concatenate([x_train, x_test])
        all_digits = all_digits.astype("float32") / 255.0
        all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
        dataset = tf.data.Dataset.from_tensor_slices(all_digits)
        dataset = dataset.shuffle(buffer_size=1024).batch(self.batch_size)

        return dataset

