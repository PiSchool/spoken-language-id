import tensorflow as tf


def base_model_fn(model, features, labels, mode, params):
    logits = model(features, training=mode == tf.estimator.ModeKeys.TRAIN, params=params)

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

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.identity(accuracy_op, name='train_acc')
        tf.summary.scalar('train_accuracy', accuracy_op)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops={'accuracy': (accuracy, accuracy_op)}
    )
