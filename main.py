import os
import json
import logging
from functools import partial

import tensorflow as tf

from models.base import base_model_fn
from models.combo import LanguidCombo
from models.rnn import LanguidRNN
from models.cnn import LanguidCNN
from models.montavon import LanguidMontavon
from data import TCData


# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='model-dir',
    default_value='model',
    docstring='Output directory for model and training stats.'
)
tf.app.flags.DEFINE_string(
    flag_name='image-dir',
    default_value=None,
    docstring='Directory with the spectrogram data.'
)
tf.app.flags.DEFINE_string(
    flag_name='train-set',
    default_value=None,
    docstring='A CSV file containing the training set: spectrogram and label pairs.'
)
tf.app.flags.DEFINE_string(
    flag_name='eval-set',
    default_value=None,
    docstring='A CSV file containing the evaluation set: spectrogram and label pairs.'
)
tf.app.flags.DEFINE_string(
    flag_name='params',
    default_value=None,
    docstring='A JSON file containing hyperparameters.'
)
tf.app.flags.DEFINE_string(
    flag_name='model',
    default_value='combo',
    docstring='Which model to use (rnn, combo).'
)
tf.app.flags.DEFINE_boolean(
    flag_name='evaluate',
    default_value=None,
    docstring='Run evaluation instead of training (and evaluation).'
)
tf.app.flags.DEFINE_string(
    flag_name='predict',
    default_value=None,
    docstring='Path to a png spectrogram for which to predict language.'
)
tf.app.flags.DEFINE_string(
    flag_name='predict-dir',
    default_value=None,
    docstring='Predict language for all png spectrograms in the specified directory.'
)
tf.app.flags.DEFINE_string(
    flag_name='model-checkpoint',
    default_value=None,
    docstring='Use the specified checkpoint file.'
)


def get_params():
    # Parameters common to all models
    common_params = dict(
        spectrogram_bins=128,
        language_count=8,
        batch_size=16,
        optimizer='momentum',
        learning_rate=0.001,
        momentum=0.9,
        eval_percent=5,
        eval_frequency=100,
        eval_steps=None,
        eval_epochs=1,
        train_epochs=40,
        shuffle_buffer_size=10000,
        language_list=[""]
    )

    # Default parameters for the CNN + RNN model
    combo_params = tf.contrib.training.HParams(
        **common_params,
        gru_num_units=128,
        dropout=0,
        pool_dropout=0,
        regularize=0,
        normalize=True,
    )

    # Default parameters for the CNN model
    cnn_params = tf.contrib.training.HParams(
        **common_params,
        spectrogram_width=858,
        dropout=0,
        pool_dropout=0,
        regularize=0,
        normalize=True,
    )

    # Default parameters for the RNN model
    rnn_params = tf.contrib.training.HParams(
        **common_params,
        gru_num_units=500,
        dropout=0,
        regularize=0,
        normalize=True,
    )

    # Default parameters for the CNN model
    montavon_params = tf.contrib.training.HParams(
        **common_params,
        spectrogram_width=858,
        dropout=0,
        pool_dropout=0,
        regularize=0,
        normalize=True,
    )

    if FLAGS.model == 'combo':
        params = combo_params
        model_fn = partial(base_model_fn, LanguidCombo)
        tf.logging.info("Running the COMBO model")
    elif FLAGS.model == 'cnn':
        params = cnn_params
        model_fn = partial(base_model_fn, LanguidCNN)
        tf.logging.info("Running the CNN model")
    elif FLAGS.model == 'rnn':
        params = rnn_params
        model_fn = partial(base_model_fn, LanguidRNN)
        tf.logging.info("Running the RNN model")
    elif FLAGS.model == 'montavon':
        params = montavon_params
        model_fn = partial(base_model_fn, LanguidMontavon)
        tf.logging.info("Running the Montavon model")

    # Parameters can be overridden from a JSON file
    if FLAGS.params:
        with open(FLAGS.params) as params_file:
            params.parse_json(params_file.read())
    tf.logging.info(params)

    run_config = tf.contrib.learn.RunConfig(
        model_dir=FLAGS.model_dir,

        # Compute a summary point once every this many steps
        save_summary_steps=100,

        # After how many steps of training checkpoints should be saved
        # Note that evaluation can only run after a new checkpoint is saved
        save_checkpoints_steps=params.eval_frequency,
    )
    return model_fn, run_config, params


def get_inputs(params, validation=False):
    """Return input function and initializer hook."""
    class IteratorInitHook(tf.train.SessionRunHook):
        def __init__(self):
            super(IteratorInitHook, self).__init__()
            self.iterator_init = None

        def after_create_session(self, session, coord):
            session.run(self.iterator_init)

    image_dir = FLAGS.image_dir
    if validation:
        epochs = params.eval_epochs
        if FLAGS.eval_set is not None:
            # Validation data is in a separate file
            tail = False
            data_file = FLAGS.eval_set
            use_percent = 100
        else:
            # Validation data is a hold-out from the test set
            tail = True
            data_file = FLAGS.train_set
            use_percent = params.eval_percent
    else:
        # Training data
        tail = False
        epochs = params.train_epochs
        data_file = FLAGS.train_set
        if FLAGS.eval_set is not None:
            # Use all data for training, validation set is in a separate file
            use_percent = 100
        else:
            # Leave some data for the validation set
            use_percent = 100 - params.eval_percent

    data = TCData(image_dir, data_file, params)
    data.load_data()
    usable_data = data.get_data(use_percent=use_percent, tail=tail)

    init_hook = IteratorInitHook()

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(usable_data)
        dataset = dataset.shuffle(params.shuffle_buffer_size)
        dataset = dataset.map(data.instance_as_tensor)
        dataset = dataset.repeat(epochs)
        dataset = dataset.padded_batch(
            params.batch_size,
            padded_shapes=([None, None, None], [])
        )

        iterator = dataset.make_initializable_iterator()
        init_hook.iterator_init = iterator.initializer
        next_example, next_label = iterator.get_next()
        return next_example, next_label

    return input_fn, init_hook


def experiment_fn(model_fn, run_config, params):
    model = tf.estimator.Estimator(model_fn, params=params, config=run_config)

    train_input_fn, train_input_hook = get_inputs(params)
    eval_input_fn, eval_input_hook = get_inputs(params, validation=True)

    class BestCheckpointHook(tf.train.CheckpointSaverHook):
        """Session hook that keeps saving the models with best accuracy so far."""
        def __init__(self, checkpoint_dir, save_secs=1, *args, **kwargs):
            super().__init__(checkpoint_dir, save_secs, checkpoint_basename='best.ckpt', *args, **kwargs)
            self.current_best = 0
            self.last_accuracy = 0
            self.last_total = 0
            self.last_count = 0
            self.last_step = 0

        def before_run(self, run_context):
            g = tf.get_default_graph()
            return tf.train.SessionRunArgs({
                'total': g.get_tensor_by_name('accuracy/total:0'),
                'count': g.get_tensor_by_name('accuracy/count:0'),
                'accuracy': g.get_tensor_by_name('accuracy/value:0'),
            })

        def after_run(self, run_context, run_values):
            self.last_accuracy = run_values.results['accuracy']
            self.last_total = run_values.results['total']
            self.last_count = run_values.results['count']

        def end(self, session):
            if self.last_accuracy > self.current_best:
                # We've finished the evaluation run and have a new best
                self.current_best = self.last_accuracy
                self._save(session, session.run(self._global_step_tensor))
                tf.logging.info("New best accuracy {:.7f} ({:.0f} / {:.0f}), saved to {}.".format(
                    self.current_best, self.last_total, self.last_count, self._save_path
                ))

            for l in self._listeners:
                l.end(session, last_step)

    best_hook = BestCheckpointHook(run_config.model_dir)

    experiment = tf.contrib.learn.Experiment(
        estimator=model,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        min_eval_frequency=params.eval_frequency,
        train_monitors=[train_input_hook],
        eval_hooks=[eval_input_hook, best_hook],
        eval_steps=params.eval_steps
    )
    return experiment


def find_best_checkpoint(run_config):
    if FLAGS.model_checkpoint:
        return os.path.join(run_config.model_dir, FLAGS.model_checkpoint)

    # Check if there is a special checkpoint with the best accuracy
    model_files = os.listdir(run_config.model_dir)
    best_checkpoints = sorted(
        [f for f in model_files if 'best.ckpt' in f and 'index' in f],
        key=lambda fname: int(''.join(filter(str.isdigit, fname)))
    )

    if best_checkpoints:
        best_checkpoint_name = os.path.splitext(best_checkpoints[-1])[0]
        return os.path.join(run_config.model_dir, best_checkpoint_name)

    return None


def predict_single(model_fn, run_config, params, png_path):
    """Predict the language given the path to a single spectrogram in png format."""
    model = tf.estimator.Estimator(model_fn, params=params, config=run_config)

    # Load data
    if not params.language_list:
        # Language order can be loaded from the training set
        data = TCData(FLAGS.image_dir, FLAGS.train_set, params)
        data.load_data()
        language_list = data.language_set
    else:
        language_list = params.language_list

    checkpoint_path = find_best_checkpoint(run_config)
    iterator = model.predict(
        input_fn=lambda: TCData.instance_as_tensor(png_path),
        checkpoint_path=checkpoint_path,
    )
    prediction = next(iterator)

    # Generate a list of (lang_id, probability) pairs
    pred_probs = [(language_list[l], p * 100) for l, p in enumerate(prediction['probs'])]

    # Sort them by probability, take 10 most likely
    pred_probs = sorted(pred_probs, key=lambda x: x[1], reverse=True)[:10]

    # Format them
    pred_probs = ["{}: {:.2f}%".format(l, p) for l, p in pred_probs]

    tf.logging.info("Predicting language for {}".format(png_path))
    tf.logging.info("Predicted language: {}".format(language_list[prediction['class']]))
    tf.logging.info("\n".join(pred_probs))


def evaluate(model_fn, run_config, params):
    # Set necessary parameters
    params.set_from_map({
        'eval_epochs': 1,
        'eval_percent': 100,
        'batch_size': 1
    })
    model = tf.estimator.Estimator(model_fn, params=params, config=run_config)
    input_fn, input_hook = get_inputs(params, validation=True)

    checkpoint_path = find_best_checkpoint(run_config)
    metrics = model.evaluate(input_fn=input_fn, hooks=[input_hook], checkpoint_path=checkpoint_path)
    tf.logging.info(metrics)


def train_or_predict(argv=None):
    model_fn, run_config, params = get_params()

    if FLAGS.evaluate:
        evaluate(model_fn, run_config, params)
    elif FLAGS.predict_dir:
        filenames = os.listdir(FLAGS.predict_dir)
        for filename in sorted(filenames):
            if filename.endswith(".png"):
                file_path = os.path.join(FLAGS.predict_dir, filename)
                predict_single(model_fn, run_config, params, file_path)
    elif FLAGS.predict:
        # Predict the label for a single sample
        predict_single(model_fn, run_config, params, FLAGS.predict)
    else:
        # Train
        tf.contrib.learn.learn_runner.run(
            experiment_fn=partial(experiment_fn, model_fn),
            run_config=run_config,
            schedule="train_and_evaluate",
            hparams=params,
        )


if __name__ == '__main__':
    # Logging and warning
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging._handler.setFormatter(logging.Formatter("%(message)s", None))

    tf.app.run(main=train_or_predict)
