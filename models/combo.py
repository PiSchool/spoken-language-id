import tensorflow as tf


def languid_combo(features, training, params):
    batch_size = tf.shape(features)[0]
    input_sgram = tf.reshape(features, [batch_size, -1, params.spectrogram_bins])

    # Switch time and freq axis
    input_reshaped = tf.transpose(input_sgram, perm=[0, 2, 1])

    # We only have a single channel
    input_reshaped = tf.expand_dims(input_reshaped, -1)

    with tf.variable_scope("CNN"):
        # TODO move CNN layer creation to a method
        conv1 = tf.layers.conv2d(input_reshaped, filters=16, kernel_size=7, activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=(2, 1), padding='same')
        if params.normalize:
            pool1 = tf.contrib.layers.layer_norm(pool1)

        conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=5, activation=tf.nn.relu)
        if params.pool_dropout:
            conv2 = tf.layers.dropout(conv2, rate=params.pool_dropout, training=training)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=3, strides=(2, 1), padding='same')
        if params.normalize:
            pool2 = tf.contrib.layers.layer_norm(pool2)

        conv3 = tf.layers.conv2d(pool2, filters=32, kernel_size=3, activation=tf.nn.relu)
        if params.pool_dropout:
            conv3 = tf.layers.dropout(conv3, rate=params.pool_dropout, training=training)
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=3, strides=(2, 1), padding='same')
        if params.normalize:
            pool3 = tf.contrib.layers.layer_norm(pool3)

        conv4 = tf.layers.conv2d(pool3, filters=32, kernel_size=3, activation=tf.nn.relu)
        if params.pool_dropout:
            conv4 = tf.layers.dropout(conv4, rate=params.pool_dropout, training=training)
        pool4 = tf.layers.max_pooling2d(conv4, pool_size=3, strides=(2, 1), padding='same')
        if params.normalize:
            pool4 = tf.contrib.layers.layer_norm(pool4)

    # The shape was (batch, bins, timesteps, filters), and should become (batch, timesteps, features)
    # TODO because of difference in pooling I get (32, 6, 844, 32) and orig. has (32, 8, 852, 32)
    input_gru = tf.transpose(pool4, perm=[0, 2, 3, 1])
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
    dense = tf.layers.dense(inputs=final_state, units=params.language_count)

    return dense
