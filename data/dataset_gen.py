import os
import csv
import math
import string
import random
import hashlib
import argparse
import itertools
from collections import Counter

import audioop
import librosa
import numpy as np

import .utils


# A list of bad quality recordings
BLACKLIST = ['Eldorplus', 'Eresus', 'Airon90-20091204-zzg-it-0849', 'estaciones_',
             'JhonattanPaniagua', 'Sam-20090404', 'trabajo_infantil', 'Marcelo-20131106']


def make_args():
    parser = argparse.ArgumentParser(description="Generate test and evaluation sets from multiple sources.")
    parser.add_argument('--per-speaker', default=20, type=int, help="Limit the number of recordings per speaker")
    parser.add_argument('-a', '--audio-dirs', nargs='+', help="Directory containing audio files (to check if entries exist, should be provided for each dataset)")
    parser.add_argument('-i', '--input-lists', nargs='+', help="A CSV file containing the list of audio file, language pairs (can be provided multiple times)")
    parser.add_argument('-c', '--input-limits', nargs='+', type=int, help="Limit the number of recordings per language in the respective dataset (can be provided multiple times)")
    parser.add_argument('--no-missing-check', default=False, action='store_true', help="Do not check if data samples actually exist")
    parser.add_argument('--mfcc', default=False, action='store_true', help="Make MFCCs instead of linear spectrograms")
    parser.add_argument('-o', '--output-dir', required=True, help="Output directory where spectrograms and CSV list files will be placed")
    parser.add_argument('-t', '--train-file', default="train_set.csv", help="Output file for training set")
    parser.add_argument('-e', '--eval-file', default="eval_set.csv", help="Output file for evaluation set")
    parser.add_argument('-s', '--eval-split', default=10, type=int, help="At least what percent of data should go to the evaluation set")
    parser.add_argument('-l', '--langs', help="Comma-separated list of language codes to include")
    parser.add_argument('-z', '--single', help="Simply convert a single audio file to a spectrogram and save in the output directory")
    parser.add_argument('-p', '--augment', default=0, type=int, help="If greater than zero, create the specified amount of perturbations")
    return parser.parse_args()


def process_input(args, audio_dir, input_filename, per_lang, languages):
    skipped = 0
    original_count = 0
    output_list = []
    lang_counter = Counter()

    with open(input_filename) as input_file:
        reader = csv.reader(input_file)
        for audio_filename, lang, speaker, count in reader:
            original_count += 1

            if not lang in languages:
                # We are not interested in this language
                continue

            if not args.no_missing_check and not os.path.isfile(os.path.join(audio_dir, audio_filename)):
                # Check if sample files actually exist, if necessary
                skipped += 1
                continue

            if int(count) > args.per_speaker:
                # We have enough samples from this speaker
                continue

            lang_counter[lang] += 1
            if lang_counter[lang] > per_lang:
                # We have enough samples of this language
                continue

            if any(banned in audio_filename for banned in BLACKLIST):
                # This filename matches a banned string
                print("Not using {} as it contains a blacklisted phrase".format(audio_filename))
                continue

            output_list.append([audio_filename, lang, speaker, audio_dir])
    return output_list, original_count, skipped


def shuffle(entries):
    """Shuffle so that all entries with the same speaker stay together."""
    salt = ''.join(random.choice(string.ascii_lowercase) for _ in range(6))
    # The key is a salted hash of the speaker
    salted_hash = lambda e: hashlib.md5((salt + e[2]).encode('utf-8')).hexdigest()
    entries.sort(key=salted_hash)
    return entries


def split(entries, at=10):
    """Split into two sets with no overlapping speakers, so that the second
    set contains at least "at" percent of data."""
    if len(entries) == 0:
        return [], []

    current_speaker = entries[0][2]
    split_position = len(entries) * at / 100.
    position = 0
    eval_set = []
    while position < split_position:
        while position < len(entries) and entries[position][2] == current_speaker:
            # Keep adding the files of the same speaker until exhausted
            eval_set.append(entries[position])
            position += 1
        if position < len(entries):
            # Move to the next speaker
            current_speaker = entries[position][2]
    return eval_set, entries[position:]


def _write_spectrogram(spectrogram, png_path):
    image = utils.spectrogram_to_image(spectrogram)
    image.save(png_path)


def generate_spectrograms(time_series, png_path, mfcc=False, verbose=False, augment=0):
    if verbose:
        print("Preparing a spectrogram {}".format(png_path))

    if mfcc:
        spectrogram = librosa.core.logamplitude(
            librosa.feature.mfcc(time_series, sr=44100, n_mfcc=20)
        )
    else:
        spectrogram = librosa.core.logamplitude(
            librosa.core.stft(time_series, window='hann', n_fft=1024, hop_length=512),
            amin=0.0008,
            ref=np.max,
        )

    # Write the original spectrogram
    _write_spectrogram(spectrogram, png_path)
    spec_names = [os.path.basename(png_path)]

    for aug_id in range(augment):
        # Perform a vocal tract length perturbation
        alpha_range = (0.9, 1.1)
        alpha = np.random.uniform(*alpha_range)
        perturbed_spectrogram = perturb(spectrogram, alpha=alpha)
        rootname, _ = os.path.splitext(png_path)
        aug_name = '{name}_aug{aug_id}.png'.format(name=rootname, aug_id=aug_id + 1)

        # Write the perturbed spectrogram
        _write_spectrogram(perturbed_spectrogram, aug_name)
        spec_names.append(os.path.basename(aug_name))

    return spec_names


def get_time_series(wave_path, trim=False):
    try:
        time_series, sampling_rate = librosa.load(wave_path, sr=44100)
    except (audioop.error, EOFError):
        print("Error reading {}".format(wave_path))
        return None, None

    if np.max(time_series) == 0:
        # This clip is just silence
        return None, None

    if trim:
        # Trim silence
        time_series, (start, end) = librosa.effects.trim(time_series)

        if end == 0:
            # This clip is just silence
            return None, None

    return time_series, sampling_rate


def perturb(spec, sr=44100, alpha=1.0, f0=0.9, fmax=1):
    """Adapted from https://github.com/YerevaNN/Spoken-language-identification/"""
    spec = np.transpose(spec)

    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)

    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    log_scaled = map(lambda x: x * alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0, scale)
    scale = np.array(list(log_scaled))
    scale *= (freqbins - 1) / max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))

    # Frequencies represented by the bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    for i in range(freqbins):
        lower = scale[i]
        upper = scale[i + 1] if i + 1 < freqbins else lower + 1

        # Look for overlaps
        for pot in range(int(np.floor(lower)), int(np.ceil(upper))):
            min_dist = min(1, upper - lower)
            overlap = max(0, min(pot + 1 - lower, upper - pot, min_dist))
            prop = overlap / (upper - lower)
            newspec[:, pot] += prop * spec[:, i]

    return np.transpose(newspec)


def generate_short_samples(args, entries, duration=5):
    """For each given filename, split it into chunks of given duration."""
    slice_list = []
    print("Generating {} second evaluation samples".format(duration))

    for audio_filename, lang, _, audio_dir in entries:
        recording, sampling_rate = get_time_series(os.path.join(audio_dir, audio_filename))
        if recording is None:
            continue

        n_full_chunks = math.floor(len(recording) / sampling_rate / duration)
        if n_full_chunks == 0:
            continue

        recording_slices = np.split(recording[:n_full_chunks * sampling_rate * duration], n_full_chunks)
        for slice_num, rec_slice in enumerate(recording_slices):
            name_base, _ = os.path.splitext(audio_filename)
            slice_name = '{0}_{2}s_{1}.png'.format(name_base, slice_num + 1, duration)
            slice_path = os.path.join(args.output_dir, slice_name)

            generate_spectrograms(rec_slice, slice_path, args.mfcc)
            print('.', end='', flush=True)

            slice_list.append([slice_name, lang])
    print()
    return slice_list


def write_output(args, train_set, eval_set):
    train_set = list(train_set)
    eval_set = list(eval_set)

    # Create output directory if it does not exist yet
    os.makedirs(args.output_dir, exist_ok=True)

    # Write the training set
    train_filename = os.path.join(args.output_dir, args.train_file)
    print("Writing the training set of {} + {} (augmented) = {} files to {}".format(
        len(train_set),
        len(train_set) * args.augment,
        len(train_set) * (1 + args.augment),
        train_filename),
    )

    rows_to_write = []
    for row in train_set:
        wave_path = os.path.join(row[3], row[0])
        spectrogram_name = '{}.png'.format(os.path.splitext(row[0])[0])
        spectrogram_path = os.path.join(args.output_dir, spectrogram_name)
        time_series, _ = get_time_series(wave_path)
        if time_series is None:
            continue
        spec_names = generate_spectrograms(time_series, spectrogram_path, args.mfcc, augment=args.augment)
        print('.' * len(spec_names), end='', flush=True)

        # We might have potentially generated more than one spectrogram per sample
        for spec_name in spec_names:
            rows_to_write.append([spec_name, row[1]])

    with open(train_filename, 'w') as train_file:
        train_csv = csv.writer(train_file)
        random.shuffle(rows_to_write)
        for row in rows_to_write:
            train_csv.writerow(row)
    print()
    print("Wrote {} files to the training set (some might have been skipped)".format(len(rows_to_write)))

    # Write the full evalutaion set
    eval_filename = os.path.join(args.output_dir, args.eval_file)
    print("Writing the full evaluation set of {} files to {}".format(len(eval_set), eval_filename))
    with open(eval_filename, 'w') as eval_file:
        eval_csv = csv.writer(eval_file)
        for row in eval_set:
            wave_path = os.path.join(row[3], row[0])
            spectrogram_name = '{}.png'.format(os.path.splitext(row[0])[0])
            spectrogram_path = os.path.join(args.output_dir, spectrogram_name)
            time_series, _ = get_time_series(wave_path)
            if time_series is None:
                continue
            generate_spectrograms(time_series, spectrogram_path, args.mfcc)
            print('.', end='', flush=True)

            eval_csv.writerow(row[:2])
    print()

    # Write the evaluation sets of exact sample length
    for duration in (3, 5, 10):
        eval_duration_set = generate_short_samples(args, eval_set, duration=duration)
        eval_basename, eval_ext = os.path.splitext(args.eval_file)
        eval_duration_filename = '{}_{}s{}'.format(eval_basename, duration, eval_ext)
        eval_duration_path = os.path.join(args.output_dir, eval_duration_filename)
        print("Writing {} second evaluation samples to {}".format(duration, eval_duration_filename))
        with open(eval_duration_path, 'w') as eval_file:
            eval_csv = csv.writer(eval_file)
            eval_duration_set = (e[:2] for e in eval_duration_set)
            eval_csv.writerows(eval_duration_set)


if __name__ == '__main__':
    args = make_args()
    code_to_lang = {
        'en': 'English',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
    }

    if args.single:
        # Convert a single audio file into a spectrogram and exit
        time_series, _ = librosa.load(args.single, sr=44100)

        spectrogram_name = '{}.png'.format(os.path.splitext(os.path.basename(args.single))[0])
        spectrogram_path = os.path.join(args.output_dir, spectrogram_name)
        generate_spectrograms(time_series, spectrogram_path, args.mfcc)
        exit()

    if len(args.audio_dirs) != len(args.input_lists):
        print("You should specifiy as many audio directories as input list files.")
        exit(1)

    if not args.input_limits:
        args.input_limits = [1000000] * len(args.input_lists)
    elif args.input_limits and len(args.input_limits) != len(args.input_lists):
        print("You should specifiy as many recording limits as input list files.")
        exit(1)

    if args.langs:
        codes = args.langs.split(',')
        languages = [lang for code, lang in code_to_lang.items() if code in codes]
    else:
        languages = code_to_lang.values()
    print("Creating a dataset for the following {} languages: {}.".format(len(languages), ', '.join(languages)))

    entries = []
    input_summaries = []
    list_dir_limits = zip(args.input_lists, args.audio_dirs, args.input_limits)
    for list_filename, audio_dir, per_lang in list_dir_limits:
        # Process each source
        print("Processing {} with audio files in {}, limiting to {} records per language".format(
            list_filename,
            audio_dir,
            per_lang
        ))
        add_entries, orig_count, skipped = process_input(args, audio_dir, list_filename, per_lang, languages)
        entries += add_entries
        input_summaries.append((list_filename, len(add_entries), orig_count, skipped))

    shuffle(entries)
    eval_set, train_set = split(entries, args.eval_split)
    write_output(args, train_set, eval_set)

    print("==== Summary ====")
    for input_summary in input_summaries:
        print("Input counts ({0}): {2}, used: {1}, skipped: {3}".format(*input_summary))
    print("Training set count ({}): {}".format(args.train_file, len(train_set)))
    print("Evaluation set count ({}): {}".format(args.eval_file, len(eval_set)))
