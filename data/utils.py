import numpy as np
from PIL import Image


def spectrogram_to_image(spectrogram):
    """Given a 2D nd array spectrogram, convert it into a grayscale image."""
    # Map to 0-255 range
    spectrogram = spectrogram[:128, :]
    spectrogram = np.abs(spectrogram)
    spectrogram -= np.min(spectrogram)
    spectrogram *= 255. / np.max(spectrogram)
    spectrogram = np.flipud(abs(255. - spectrogram))

    # Write the image to a file
    image = Image.fromarray(spectrogram[-128:, :])
    image = image.convert('L')
    return image
