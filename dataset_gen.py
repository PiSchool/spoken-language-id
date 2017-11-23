import os
import csv
import string
import random
import hashlib
import argparse
import itertools
from collections import Counter

from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


def make_args():
    parser = argparse.ArgumentParser(description="Generate test and evaluation sets from multiple sources.")
    parser.add_argument('--per-speaker', default=20, type=int, help="Limit the number of recordings per speaker")
    parser.add_argument('-a', '--audio-dir', help="Directory containing audio files (to check if entries exist)")
    parser.add_argument('--voxforge-file', help="VoxForge data CSV file")
    parser.add_argument('--voxforge-per-lang', default=1250, type=int, help="Limit the number of recordings per language for VoxForge")
    parser.add_argument('--audiolingua-file', help="Audio Lingua data CSV file")
    parser.add_argument('--audiolingua-per-lang', default=210, type=int, help="Limit the number of recordings per language for Audio Lingua")
    parser.add_argument('--no-missing-check', default=False, action='store_true', help="Do not check if data samples actually exist")
    parser.add_argument('-t', '--train-file', required=True, help="Output file for training set")
    parser.add_argument('-e', '--eval-file', required=True, help="Output file for evaluation set")
    parser.add_argument('-s', '--eval-split', default=10, type=int, help="At least what percent of data should go to the evaluation set.")
    return parser.parse_args()


def process_input(args, input_filename, per_lang):
    skipped = 0
    original_count = 0
    output_list = []
    lang_counter = Counter()

    with open(input_filename) as input_file:
        reader = csv.reader(input_file)
        for audio_filename, lang, speaker, count in reader:
            original_count += 1

            if not args.no_missing_check and not os.path.isfile(os.path.join(args.audio_dir, audio_filename)):
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

            output_list.append([audio_filename, lang, speaker])
    return output_list, original_count, skipped


def shuffle(entries):
    """Shuffle so that all entries with the same speaker stay together."""
    salt = ''.join(random.choices(string.ascii_lowercase, k=5))
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


def generate_short_samples(args, entries, duration=5):
    """For each given filename, split it into chunks of given duration."""
    slice_list = []
    print("Generating {} second evaluation samples".format(duration))
    for audio_filename, lang in entries:
        try:
            recording = AudioSegment.from_mp3(os.path.join(args.audio_dir, audio_filename))
        except CouldntDecodeError:
            print("Error decoding {}".format(audio_filename))
            continue

        recording_slices = recording[::duration * 1000]
        for slice_num, rec_slice in enumerate(recording_slices):
            name_base, name_ext = os.path.splitext(audio_filename)
            file_format = name_ext.strip('.')
            slice_name = '{0}_{3}s_{2}{1}'.format(name_base, name_ext, slice_num + 1, duration)
            slice_path = os.path.join(args.audio_dir, slice_name)

            if len(rec_slice) == duration * 1000:
                # Only save slices of the desired duration
                rec_slice.export(slice_path, format=file_format)
                slice_list.append([slice_name, lang])
    return slice_list


def write_output(args, train_set, eval_set):
    # Only leave the first two columns
    train_set = (e[:2] for e in train_set)
    eval_set = (e[:2] for e in eval_set)

    # Write the training set
    print("Writing training set to {}".format(args.train_file))
    with open(args.train_file, 'w') as train_file:
        train_csv = csv.writer(train_file)
        train_csv.writerows(train_set)

    # Write the full evalutaion set
    eval_set = list(eval_set)
    print("Writing full evaluation set to {}".format(args.eval_file))
    with open(args.eval_file, 'w') as eval_file:
        eval_csv = csv.writer(eval_file)
        eval_csv.writerows(eval_set)

    # Write the evaluation sets of exact sample length
    for duration in (3, 5, 10):
        eval_duration_set = generate_short_samples(args, eval_set, duration=duration)
        eval_basename, eval_ext = os.path.splitext(args.eval_file)
        eval_duration_filename = '{}_{}s{}'.format(eval_basename, duration, eval_ext)
        print("Writing {} second evaluation samples to {}".format(duration, eval_duration_filename))
        with open(eval_duration_filename, 'w') as eval_file:
            eval_csv = csv.writer(eval_file)
            eval_csv.writerows(eval_duration_set)


if __name__ == '__main__':
    args = make_args()
    languages = ["English", "French", "German", "Italian", "Portuguese", "Spanish"]

    entries = []
    file_data = []
    if args.voxforge_file:
        add_entries, orig_count, skipped = process_input(args, args.voxforge_file, args.voxforge_per_lang)
        entries += add_entries
        file_data.append((args.voxforge_file, len(add_entries), orig_count, skipped))

    if args.audiolingua_file:
        add_entries, orig_count, skipped = process_input(args, args.audiolingua_file, args.audiolingua_per_lang)
        entries += add_entries
        file_data.append((args.audiolingua_file, len(add_entries), orig_count, skipped))

    shuffle(entries)
    eval_set, train_set = split(entries, args.eval_split)
    write_output(args, train_set, eval_set)

    print("==== Summary ====")
    for input_summary in file_data:
        print("Input counts ({0}): {2}, used: {1}, skipped: {3}".format(*input_summary))
    print("Training set count ({}): {}".format(args.train_file, len(train_set)))
    print("Evaluation set count ({}): {}".format(args.eval_file, len(eval_set)))
