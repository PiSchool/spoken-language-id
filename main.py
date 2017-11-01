import json
import argparse
import logging as log

import tensorflow as tf

from models.rnn import LanguidRNN
from data import TCData


def make_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-c', '--conf', help="JSON file with configuration")
    arg_parser.add_argument('-l', '--load', help="Load an existing model")
    return arg_parser.parse_args()


if __name__ == '__main__':
    conf = {
        'save_every': 1,
        'print_every': 1,
        'spectrogram_bins': 128,
        'language_count': 176,
        'gru_num_units': 500,
        'batch_size': 32,
        'learning_rate': 0.003,
        'momentum': 0.9,
        'epochs': 4,
        'save_summary': 'summary',
        'save_path': 'model',
    }

    # Logging and warnings
    log.basicConfig(format='%(message)s', level=log.INFO)
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Override parameters using external configuration
    args = make_args()
    if args.conf:
        with open(args.conf, 'r') as config_file:
            external_config = json.load(config_file)
            conf.update(external_config)

    if args.load:
        conf['load_path'] = args.load

    if not 'png_dir' in conf or not 'label_filename' in conf:
        exit("Please provide input sources in the configuration file.")

    # Load data
    data = TCData(conf)
    data.load_data()

    # Create and train the model
    languid = LanguidRNN(conf, data, training=True)

    if conf.get('load_path'):
        languid.load()

    languid.train(data)
    languid.close()
