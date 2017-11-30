import tensorflow as tf


class LanguidRNN:
    def __init__(self, features, training, params):
        # A sequence of spectrograms representing a sample audio, (batch_size, timesteps, bins)
        batch_size = tf.shape(features)[0]
        input_sgram = tf.reshape(features, [batch_size, -1, params.spectrogram_bins])
        normal_input = tf.subtract(tf.multiply(input_sgram, 2), 1)

        with tf.variable_scope("GRU1"):
            gru_cell = tf.contrib.rnn.GRUCell(num_units=params.gru_num_units)
            output_gru, final_state = tf.nn.dynamic_rnn(gru_cell, normal_input, dtype=tf.float32)

            if params.normalize:
                # Optional layer normalization
                output_gru = tf.contrib.layers.layer_norm(output_gru)

        with tf.variable_scope("GRU2"):
            gru_cell = tf.contrib.rnn.GRUCell(num_units=params.gru_num_units)
            output_gru, final_state = tf.nn.dynamic_rnn(gru_cell, output_gru, dtype=tf.float32)

            if params.normalize:
                # Optional layer normalization
                final_state = tf.contrib.layers.layer_norm(final_state)

            if params.dropout:
                # Optional dropout
                final_state = tf.layers.dropout(final_state, rate=params.dropout, training=training)

        # The prediction layer
        self.dense = tf.layers.dense(inputs=final_state, units=params.language_count)

    def get_prediction_logits(self):
        return self.dense
