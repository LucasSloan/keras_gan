import os

import tensorflow as tf
import save_images

class SamplingCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, batch_size, latent_dim):
        super().__init__()

        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        os.makedirs(f'{self.checkpoint_dir}/samples')

    def on_epoch_end(self, epoch, logs):
        epoch_one_indexed = epoch + 1
        noise = tf.random.normal(shape=(self.batch_size, self.latent_dim))

        imgs = self.model({'noise': noise})

        save_images.save_images(imgs, [8, 8], f'{self.checkpoint_dir}/samples/epoch_{epoch_one_indexed}.png', range_min=0.0)