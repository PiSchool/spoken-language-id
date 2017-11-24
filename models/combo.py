import tensorflow as tf


def convpool_layers(inputs, training, filters, kernel_size, pool_size, strides, normalize, pool_dropout):
    conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, activation=tf.nn.relu)
    if pool_dropout:
        conv = tf.layers.dropout(conv, rate=pool_dropout, training=training)
    pool = tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=strides, padding='same')
    if normalize:
        pool = tf.contrib.layers.layer_norm(pool)
    return pool


def languid_combo(features, training, params):
    batch_size = tf.shape(features)[0]
    input_sgram = tf.reshape(features, [batch_size, -1, params.spectrogram_bins])

    # Switch time and freq axis
    input_reshaped = tf.transpose(input_sgram, perm=[0, 2, 1])

    # We only have a single channel
    input_reshaped = tf.expand_dims(input_reshaped, -1)

    with tf.variable_scope("CNN"):
        convpool = convpool_layers(
            input_reshaped, training, filters=16, kernel_size=7, pool_size=3, strides=(2, 1),
            normalize=params.normalize, pool_dropout=params.pool_dropout,
        )
        convpool = convpool_layers(
            convpool, training, filters=32, kernel_size=5, pool_size=3, strides=(2, 1),
            normalize=params.normalize, pool_dropout=params.pool_dropout,
        )
        convpool = convpool_layers(
            convpool, training, filters=32, kernel_size=3, pool_size=3, strides=(2, 1),
            normalize=params.normalize, pool_dropout=params.pool_dropout,
        )
        convpool = convpool_layers(
            convpool, training, filters=32, kernel_size=3, pool_size=3, strides=(2, 1),
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
    dense = tf.layers.dense(inputs=final_state, units=params.language_count)

    return dense
