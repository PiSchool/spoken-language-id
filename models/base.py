import tensorflow as tf


class CNNBaseModel:
    def add_convpool_layer(self, inputs, training, filters, kernel_size,
                    pool_size, pool_strides, normalize, pool_dropout,
                    double_padding=False):
        """A helper function for defining convolutional layers followed my max-pooling."""
        padding = 'same'
        if double_padding:
            # Perform manual padding, because 'same' only pads by one
            padding = 'valid'
            inputs = tf.pad(inputs, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))

        # Convolutional layer
        conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, activation=tf.nn.relu)
        if pool_dropout:
            conv = tf.layers.dropout(conv, rate=pool_dropout, training=training)

        # Pooling layer
        pool = tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=pool_strides, padding=padding)
        if normalize:
            pool = tf.contrib.layers.layer_norm(pool)
        return pool


def base_model_fn(model_class, features, labels, mode, params):
    model = model_class(features, training=mode == tf.estimator.ModeKeys.TRAIN, params=params)
    logits = model.get_prediction_logits()

    # The prediction
    pred_classes = tf.argmax(logits, axis=-1)
    pred_probabilities = tf.nn.softmax(logits)
    predictions = {
        'class': pred_classes,
        'probs': pred_probabilities,
    }

    # If predicting, no need to define loss etc.
    if mode == tf.estimator.ModeKeys.PREDICT:
        batch_size = tf.shape(features['sgram'])[0]
        lang_list = tf.reshape(
            tf.tile(
                tf.constant(params.language_list, shape=[params.language_count], dtype=tf.string),
                multiples=[batch_size]
            ),
            [batch_size, -1]
        )
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs={
                'language': tf.estimator.export.ClassificationOutput(
                    scores=pred_probabilities,
                    classes=lang_list,
                )
            }
        )

    onehot_labels = tf.one_hot(labels, depth=params.language_count)
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)
    if params.regularize:
        # Add L2 regularization to non-bias variables if configured
        non_bias_vars = [v for v in tf.trainable_variables() if not 'bias' in v.name]
        l2_regularizer = tf.contrib.layers.l2_regularizer(params.regularize)
        loss += tf.contrib.layers.apply_regularization(
            l2_regularizer, non_bias_vars
        )

    if params.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate,
        )
    else:
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=params.learning_rate,
            momentum=params.momentum
        )

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    accuracy, accuracy_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # Compute the confusion matrix
    with tf.variable_scope("confusion"):
        confusion_matrix = tf.confusion_matrix(
            labels=labels, predictions=pred_classes, num_classes=params.language_count
        )
        confusion_matrix_total = tf.Variable(
            tf.zeros(shape=(params.language_count, params.language_count), dtype=tf.int32),
            trainable=False,
            name="confusion_matrix_total",
            collections=[tf.GraphKeys.LOCAL_VARIABLES]
        )
        confusion_update_op = tf.assign_add(confusion_matrix_total, confusion_matrix)
        confusion_matrix_total = tf.convert_to_tensor(confusion_matrix_total)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.identity(accuracy_op, name='train_acc')
        tf.summary.scalar('train_accuracy', accuracy_op)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={
            'accuracy': (accuracy, accuracy_op),
            'confusion': (confusion_matrix_total, confusion_update_op)
        }
    )
