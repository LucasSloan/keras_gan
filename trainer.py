import time

import tensorflow as tf
import numpy as np

import nn
from sampling_callback import SamplingCallback
from gan import GAN
from dataset import MnistDataset

LATENT_DIM = 128
BATCH_SIZE = 64

DISCRIMINATOR_LEARNING_RATE = 3e-4
GENERATOR_LEARNING_RATE = 3e-4

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Directory to load model state from to resume training.")
flags.DEFINE_string("experiment_name", None, "Name of the experiment being run.")
flags.DEFINE_bool("use_mixed_precision", False, "Whether to use float16 mixed precision training.")

if FLAGS.use_mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

dataset = MnistDataset(BATCH_SIZE)

discriminator = nn.get_discriminator()
generator = nn.get_generator(LATENT_DIM)

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=DISCRIMINATOR_LEARNING_RATE),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=GENERATOR_LEARNING_RATE),
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
)

if FLAGS.checkpoint_dir:
    checkpoint_dir = FLAGS.checkpoint_dir
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    gan.load_weights(latest)
elif FLAGS.experiment_name:
    checkpoint_dir = 'checkpoints/{}'.format(FLAGS.experiment_name)
else:
    checkpoint_dir = 'checkpoints/{}'.format(time.strftime("%m_%d_%y-%H_%M"))

sampling_callback = SamplingCallback(checkpoint_dir, batch_size=BATCH_SIZE, latent_dim=LATENT_DIM)

gan.fit(dataset.load(), epochs=20, callbacks=[sampling_callback])
