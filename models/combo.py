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
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=3, strides=(2, 1), padding='same')
        if params.normalize:
            pool2 = tf.contrib.layers.layer_norm(pool2)

        conv3 = tf.layers.conv2d(pool2, filters=32, kernel_size=3, activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv3, pool_size=3, strides=(2, 1), padding='same')
        if params.normalize:
            pool3 = tf.contrib.layers.layer_norm(pool3)

        conv4 = tf.layers.conv2d(pool3, filters=32, kernel_size=3, activation=tf.nn.relu)
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
        norm_output_gru = tf.layers.batch_normalization(final_state, training=training)

        if params.dropout:
            norm_output_gru = tf.layers.dropout(final_state, rate=params.dropout, training=training)

    # The prediction layer
    dense = tf.layers.dense(inputs=norm_output_gru, units=params.language_count)

    return dense


def model_fn(features, labels, mode, params):
    logits = languid_combo(features, training=mode == tf.estimator.ModeKeys.TRAIN, params=params)

    # The prediction
    pred_classes = tf.argmax(logits, axis=-1)
    pred_probabilities = tf.nn.softmax(logits)
    predictions = {
        'class': pred_classes,
        'probs': pred_probabilities,
    }

    # If predicting, no need to define loss etc.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    onehot_labels = tf.one_hot(labels, depth=params.language_count)
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)
    if params.regularize:
        # Add L2 regularization if configured
        l2_regularizer = tf.contrib.layers.l2_regularizer(params.regularize)
        loss += tf.contrib.layers.apply_regularization(
            l2_regularizer, tf.trainable_variables()
        )

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=params.learning_rate,
        momentum=params.momentum
    ).minimize(loss, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    accuracy = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('train_loss', loss)
        tf.summary.scalar('train_accuracy', tf.reduce_mean(accuracy))

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=optimizer,
        eval_metric_ops={'accuracy': accuracy}
    )
