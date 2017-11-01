import os
import csv

import numpy as np
from PIL import Image


class TCData(object):
    def __init__(self, conf):
        self.png_dir = conf['png_dir']
        self.label_filename = conf['label_filename']
        self.batch_size = conf['batch_size']
        self.language_set = []
        self.labels = []
        self.images = []

    def load_data(self):
        with open(self.label_filename) as label_file:
            csv_reader = csv.reader(label_file, delimiter=',')

            # Skip the header
            next(csv_reader)

            for image, label in csv_reader:
                self.images.append(image)
                self.labels.append(label)

        self.language_set = sorted(set(self.labels))

    @property
    def label_count(self):
        return len(self.language_set)

    @property
    def sample_count(self):
        return len(self.images)

    def lang_index(self, lang):
        """Return the index of the given language."""
        return self.language_set.index(lang)

    def get_limited_data(self, use_percent, tail=False):
        """Return the first (or the last) use_percent percent of data and
        its labels.
        """
        start = None
        end = int(round(len(self.images) * use_percent / 100.))
        if tail:
            start, end = -end, start
        return self.images[start:end], self.labels[start:end]

    def make_batches(self, use_percent=100):
        """Generator for producing batches that look like this:
        [['000.png', '001.png'], [[0, 0, 1], [0, 1, 0]]]
        """
        batch_start = 0

        # Limit to use_percent percent of input data
        usable_images, usable_labels = self.get_limited_data(use_percent)

        while batch_start < len(usable_images):
            batch_end = min(batch_start + self.batch_size, len(usable_images) - 1)
            images = []
            for audio_name in usable_images[batch_start:batch_end]:
                # Load spectrograms as images
                image_name = '{}.png'.format(audio_name.split('.')[0])
                image = Image.open(os.path.join(self.png_dir, image_name))
                images.append(np.transpose(np.array(image)[:128, :858]) / 256.)
            labels = []
            for label in usable_labels[batch_start:batch_end]:
                # Convert strings to one-hot vectors
                one_hot = np.zeros(len(self.language_set))
                one_hot[self.lang_index(label)] = 1
                labels.append(one_hot)
            yield images, labels

            batch_start += self.batch_size
