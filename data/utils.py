import librosa
import numpy as np
from PIL import Image


def series_to_spectrogram(time_series, mfcc=False):
    """Given a time series, generate a log-scale spectrogram."""
    if mfcc:
        return librosa.core.logamplitude(
            librosa.feature.mfcc(time_series, sr=44100, n_mfcc=20)
        )

    return librosa.core.logamplitude(
        librosa.core.stft(time_series, window='hann', n_fft=1024, hop_length=512),
        amin=0.0008,
        ref=np.max,
    )


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
