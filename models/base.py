import os
import logging as log

import tensorflow as tf


class BaseModel:
    """Abstract model."""

    def __init__(self, conf, name=None):
        self.session = tf.Session()
        self.name = name or self.__class__.__name__
        self.conf = conf
        self.initialized = False

    def save(self):
        """Save the model parameters."""
        model_path = self.conf.get('save_path')
        if model_path:
            save_path = os.path.join(model_path, self.name)
            log.info("Saving {} model to {}".format(self.name, save_path))
            saver = tf.train.Saver()
            saver.save(self.session, save_path)

    def load(self):
        """Load the model parameters."""
        model_path = self.conf['load_path']
        if model_path:
            load_path = os.path.join(model_path, self.name)
            saver = tf.train.Saver()

            # Initialize in case we're only loading part of parameters
            self.initialize_if_fresh()

            log.info("Loading {} model from {}".format(self.name, load_path))
            saver.restore(self.session, load_path)

    def initialize_if_fresh(self):
        if not self.initialized:
            log.info("Initializing...")
            self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        self.initialized = True

    def train(self):
        raise NotImplementedError

    def close(self):
        self.session.close()
