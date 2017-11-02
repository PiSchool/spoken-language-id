import os
import time
import logging as log

import tensorflow as tf
import numpy as np

from .base import BaseModel


class LanguidRNN(BaseModel):
    def __init__(self, conf, data, training=True):
        super().__init__(conf)

        # A sequence of spectrograms representing a sample audio, (batch_size, timesteps, bins)
        self.input_sgram = tf.placeholder(tf.float32, shape=[None, 858, self.conf['spectrogram_bins']])
        normal_input = tf.subtract(tf.multiply(self.input_sgram, 2), 1)

        # One-hot vector representing the target language (batch_size, language_count)
        self.output_lang = tf.placeholder(tf.int8, shape=[None, self.conf['language_count']])

        with tf.variable_scope("GRU1"):
            gru_cell = tf.contrib.rnn.GRUCell(num_units=self.conf['gru_num_units'])
            self.output_gru, self.final_state = tf.nn.dynamic_rnn(gru_cell, normal_input, dtype=tf.float32)
            # TODO should it be axis=1?
            norm_output_gru = tf.layers.batch_normalization(self.output_gru, training=training, axis=1)

        with tf.variable_scope("GRU2"):
            gru_cell = tf.contrib.rnn.GRUCell(num_units=self.conf['gru_num_units'])
            self.output_gru, self.final_state = tf.nn.dynamic_rnn(gru_cell, norm_output_gru, dtype=tf.float32)
            norm_output_gru = tf.layers.batch_normalization(self.final_state, training=training)

        # The prediction layer
        dense = tf.layers.dense(inputs=norm_output_gru, units=data.label_count)

        # TODO should we add l2 regularization?
        self.loss = tf.losses.softmax_cross_entropy(self.output_lang, dense)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
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
