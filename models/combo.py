import time
import logging as log

import tensorflow as tf

from .base import BaseModel


class LanguidCombo(BaseModel):
    def __init__(self, conf, data, training=True):
        super().__init__(conf)

        self.initialized = False

        # A sequence of spectrograms representing a sample audio, (batch_size, timesteps, bins)
        self.input_sgram = tf.placeholder(
            tf.float32,
            shape=[self.conf['batch_size'], self.conf['spectrogram_width'], self.conf['spectrogram_bins']]
        )

        # One-hot vector representing the target language (batch_size, language_count)
        self.output_lang = tf.placeholder(tf.int8, shape=[self.conf['batch_size'], self.conf['language_count']])

        # Switch time and freq axis
        input_reshaped = tf.transpose(self.input_sgram, perm=[0, 2, 1])

        # We only have a single channel
        input_reshaped = tf.expand_dims(input_reshaped, -1)

        with tf.variable_scope("CNN"):
            # TODO move CNN layer creation to a method
            conv1 = tf.layers.conv2d(input_reshaped, filters=16, kernel_size=7, activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=(2, 1), padding='same')
            norm_pool1 = tf.layers.batch_normalization(pool1, training=training)

            conv2 = tf.layers.conv2d(norm_pool1, filters=32, kernel_size=5, activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, pool_size=3, strides=(2, 1), padding='same')
            norm_pool2 = tf.layers.batch_normalization(pool2, training=training)

            conv3 = tf.layers.conv2d(norm_pool2, filters=32, kernel_size=3, activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(conv3, pool_size=3, strides=(2, 1), padding='same')
            norm_pool3 = tf.layers.batch_normalization(pool3, training=training)

            conv4 = tf.layers.conv2d(norm_pool3, filters=32, kernel_size=3, activation=tf.nn.relu)
            pool4 = tf.layers.max_pooling2d(conv4, pool_size=3, strides=(2, 1), padding='same')
            norm_pool4 = tf.layers.batch_normalization(pool4, training=training)

        # The shape was (batch, bins, timesteps, filters), and should become (batch, timesteps, features)
        # TODO because of difference in pooling I get (32, 6, 844, 32) and orig. has (32, 8, 852, 32)
        input_gru = tf.transpose(norm_pool4, perm=[0, 2, 3, 1])
        input_shape = tf.shape(input_gru)
        input_gru = tf.reshape(input_gru, [self.conf['batch_size'], input_gru.get_shape()[1], tf.reduce_prod(input_shape[2:])])

        with tf.variable_scope("GRU"):
            gru_cell = tf.contrib.rnn.GRUCell(num_units=self.conf['gru_num_units'])
            output_gru, final_state = tf.nn.dynamic_rnn(gru_cell, input_gru, dtype=tf.float32)
            norm_output_gru = tf.layers.batch_normalization(final_state, training=training)
            # TODO dropout?

        # The prediction layer
        dense = tf.layers.dense(inputs=norm_output_gru, units=data.label_count)

        self.loss = tf.losses.softmax_cross_entropy(self.output_lang, dense)
        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate=self.conf['learning_rate'],
            momentum=self.conf['momentum']
        ).minimize(self.loss)

        # Metrics
        self.accuracy = tf.metrics.accuracy(
            labels=tf.argmax(self.output_lang, 0),
            predictions=tf.argmax(dense, 0)
        )
        self.train_summary = tf.summary.scalar('train_loss', self.loss)
        self.train_writer = None
        if self.conf['save_summary']:
            self.train_writer = tf.summary.FileWriter(self.conf['save_summary'],
                                                      self.session.graph)

    def train(self, data):
        self.initialize_if_fresh()

        avg_loss = 0
        iteration = 0
        time_measurement = time.time()
        for epoch in range(self.conf['epochs']):
            log.info("=== Epoch {} ===".format(epoch + 1))
            for image_batch, label_batch in data.make_batches():

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                feed_dict = {
                    self.input_sgram: image_batch,
                    self.output_lang: label_batch
                }
                train_summary, _, loss, accuracy, _ = self.session.run(
                    [self.train_summary, self.optimizer, self.loss, self.accuracy, update_ops],
                    feed_dict=feed_dict
                )

                iteration += 1

                if not avg_loss:
                    avg_loss = loss
                else:
                    avg_loss = avg_loss * 0.9 + loss * 0.1

                if iteration % self.conf['print_every'] == 0:
                    # Write to a summary file
                    if self.train_writer:
                        self.train_writer.add_summary(train_summary, iteration)

                    # Print to screen
                    log.info("Processed: {} / {}, Loss: {:.3f}, Avg. loss: {:.3f}, Accuracy: {:.4f}, Duration: {:.2f}s".format(
                        iteration * data.batch_size,
                        data.sample_count,
                        loss,
                        avg_loss,
                        accuracy[1],
                        time.time() - time_measurement,
                    ))

                    time_measurement = time.time()

                if iteration % self.conf['save_every'] == 0:
                    self.save()
