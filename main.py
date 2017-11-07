import json
from functools import partial

import tensorflow as tf

from models.combo import model_fn as languid_combo_model_fn
from models.rnn import model_fn as languid_rnn_model_fn
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
    flag_name='label-file',
    default_value=None,
    docstring='A CSV file containing the labels for each spectrogram.'
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


def get_inputs(image_dir, label_file, params, validation=False):
    """Return input function and initializer hook."""
    class IteratorInitHook(tf.train.SessionRunHook):
        def __init__(self):
            super(IteratorInitHook, self).__init__()
            self.iterator_init = None

        def after_create_session(self, session, coord):
            session.run(self.iterator_init)

    # Load data
    data = TCData(image_dir, label_file, params)
    data.load_data()

    init_hook = IteratorInitHook()

    def input_parser(image_name, label):
        img_file = tf.read_file(image_name)
        image = tf.image.decode_png(img_file, channels=0)
        image = tf.cast(image, tf.float32)
        image_data = tf.transpose(image[:128, :858] / 256)
        label = tf.cast(label, tf.int32)
        return image_data, label

    def input_fn():
        # Leave some data for the validation set
        tail = False
        use_percent = 100 - params.eval_percent
        epochs = params.train_epochs
        if validation:
            # Use part of the training set for validation (take samples from the end)
            tail = True
            use_percent = params.eval_percent
            epochs = params.eval_epochs

        usable_data = data.get_data(use_percent=use_percent, tail=tail)
        dataset = tf.data.Dataset.from_tensor_slices(usable_data)
        dataset = dataset.map(input_parser)
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(params.batch_size)

        iterator = dataset.make_initializable_iterator()
        init_hook.iterator_init = iterator.initializer
        next_example, next_label = iterator.get_next()
        return next_example, next_label

    return input_fn, init_hook


def experiment_fn(model_fn, run_config, params):
    model = tf.estimator.Estimator(model_fn, params=params, config=run_config)

    train_input_fn, train_input_hook = get_inputs(FLAGS.image_dir, FLAGS.label_file, params)
    eval_input_fn, eval_input_hook = get_inputs(FLAGS.image_dir, FLAGS.label_file, params, validation=True)

    experiment = tf.contrib.learn.Experiment(
        estimator=model,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        min_eval_frequency=params.eval_frequency,
        train_monitors=[train_input_hook],
        eval_hooks=[eval_input_hook],
        eval_steps=params.eval_steps
    )
    return experiment


def run_experiment(argv=None):
    # Parameters common to all models
    common_params = dict(
        spectrogram_bins=128,
        spectrogram_width=858,
        language_count=176,
        batch_size=16,
        learning_rate=0.003,
        momentum=0.9,
        eval_percent=5,
        eval_frequency=100,
        eval_steps=None,
        eval_epochs=1,
        train_epochs=40,
    )

    # Default parameters for the CNN + RNN model
    combo_params = tf.contrib.training.HParams(
        **common_params,
        gru_num_units=128,
        dropout=0,
        regularize=0,
    )

    # Default parameters for the RNN model
    rnn_params = tf.contrib.training.HParams(
        **common_params,
        gru_num_units=500,
        dropout=0,
        regularize=0,
    )

    if FLAGS.model == 'combo':
        params = combo_params
        model_fn = languid_combo_model_fn
        tf.logging.info("Running the COMBO model")
    else:
        params = rnn_params
        model_fn = languid_rnn_model_fn
        tf.logging.info("Running the RNN model")

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

    tf.contrib.learn.learn_runner.run(
        experiment_fn=partial(experiment_fn, model_fn),
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=params,
    )
    

if __name__ == '__main__':
    # Logging and warning
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run(main=run_experiment)
