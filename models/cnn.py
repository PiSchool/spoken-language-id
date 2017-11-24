import tensorflow as tf

from .base import convpool_layers


def languid_cnn(features, training, params):
    batch_size = tf.shape(features)[0]

    # Limit the width
    features = features[:, :, :params.spectrogram_width, :]
    input_sgram = tf.reshape(features, [batch_size, params.spectrogram_width, params.spectrogram_bins])

    # Switch time and freq axis
    input_reshaped = tf.transpose(input_sgram, perm=[0, 2, 1])

    # We only have a single channel
    input_reshaped = tf.expand_dims(input_reshaped, -1)

    with tf.variable_scope("CNN"):
        convpool = convpool_layers(
            input_reshaped, training, filters=16, kernel_size=7, pool_size=3, pool_strides=2,
            double_padding=True, normalize=params.normalize, pool_dropout=params.pool_dropout,
        )
        convpool = convpool_layers(
            convpool, training, filters=32, kernel_size=5, pool_size=3, pool_strides=2,
            double_padding=True, normalize=params.normalize, pool_dropout=params.pool_dropout,
        )
        convpool = convpool_layers(
            convpool, training, filters=64, kernel_size=3, pool_size=3, pool_strides=2,
            double_padding=True, normalize=params.normalize, pool_dropout=params.pool_dropout,
        )
        convpool = convpool_layers(
            convpool, training, filters=128, kernel_size=3, pool_size=3, pool_strides=2,
            double_padding=True, normalize=params.normalize, pool_dropout=params.pool_dropout,
        )
        convpool = convpool_layers(
            convpool, training, filters=128, kernel_size=3, pool_size=3, pool_strides=2,
            double_padding=True, normalize=params.normalize, pool_dropout=params.pool_dropout,
        )
        convpool = convpool_layers(
            convpool, training, filters=256, kernel_size=3, pool_size=3, pool_strides=(3, 2),
            double_padding=True, normalize=params.normalize, pool_dropout=params.pool_dropout,
        )

    # An extra dense layer
    convpool = tf.contrib.layers.flatten(convpool)
    dense = tf.layers.dense(inputs=convpool, units=1024, activation=tf.nn.relu)
    if params.normalize:
        dense = tf.contrib.layers.layer_norm(dense)
    if params.dropout:
        dense = tf.layers.dropout(dense, rate=params.dropout, training=training)

    # The prediction layer
    predictions = tf.layers.dense(inputs=dense, units=params.language_count)

    return predictions
