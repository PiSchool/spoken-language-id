import tensorflow as tf

from .base import CNNBaseModel

class LanguidCombo(CNNBaseModel):
    def __init__(self, features, training, params):
        batch_size = tf.shape(features)[0]
        input_sgram = tf.reshape(features, [batch_size, -1, params.spectrogram_bins])

        # Smaller kernels for less spectrogram bins (e.g. MFCC)
        if params.spectrogram_bins < 30:
            kernels = [4, 2, 2, 2]
            pool_strides = [(2, 1), 1, 1, 1]
        else:
            kernels = [7, 5, 3, 3]
            pool_strides = [(2, 1), (2, 1), (2, 1), (2, 1)]

        # Switch time and freq axis
        input_reshaped = tf.transpose(input_sgram, perm=[0, 2, 1])

        # We only have a single channel
        input_reshaped = tf.expand_dims(input_reshaped, -1)

        with tf.variable_scope("CNN"):
            convpool = self.add_convpool_layer(
                input_reshaped, training, filters=16, kernel_size=kernels[0], pool_size=3, pool_strides=pool_strides[0],
                normalize=params.normalize, pool_dropout=params.pool_dropout,
            )
            convpool = self.add_convpool_layer(
                convpool, training, filters=32, kernel_size=kernels[1], pool_size=3, pool_strides=pool_strides[1],
                normalize=params.normalize, pool_dropout=params.pool_dropout,
            )
            convpool = self.add_convpool_layer(
                convpool, training, filters=32, kernel_size=kernels[2], pool_size=3, pool_strides=pool_strides[2],
                normalize=params.normalize, pool_dropout=params.pool_dropout,
            )
            convpool = self.add_convpool_layer(
                convpool, training, filters=32, kernel_size=kernels[3], pool_size=3, pool_strides=pool_strides[3],
                normalize=params.normalize, pool_dropout=params.pool_dropout,
            )

        # The shape was (batch, bins, timesteps, filters), and should become (batch, timesteps, features)
        # TODO because of difference in pooling I get (32, 6, 844, 32) and orig. has (32, 8, 852, 32)
        input_gru = tf.transpose(convpool, perm=[0, 2, 3, 1])
        input_shape = input_gru.get_shape()
        input_gru = tf.reshape(input_gru, tf.stack([batch_size, -1, input_shape[2] * input_shape[3]]))

        with tf.variable_scope("GRU"):
            gru_cell = tf.contrib.rnn.GRUCell(num_units=params.gru_num_units)
            output_gru, final_state = tf.nn.dynamic_rnn(gru_cell, input_gru, dtype=tf.float32)
            if params.normalize:
                final_state = tf.contrib.layers.layer_norm(final_state)
            if params.dropout:
                final_state = tf.layers.dropout(final_state, rate=params.dropout, training=training)

        # The prediction layer
        self.dense = tf.layers.dense(inputs=final_state, units=params.language_count)

    def get_prediction_logits(self):
        return self.dense
