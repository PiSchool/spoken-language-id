import os
import csv
import string
import random
import hashlib
import argparse
from collections import Counter


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


def write_output(args, train_set, eval_set):
    # Only leave the first two columns
    train_set = (e[:2] for e in train_set)
    eval_set = (e[:2] for e in eval_set)

    with open(args.train_file, 'w') as train_file:
        train_csv = csv.writer(train_file)
        train_csv.writerows(train_set)

    with open(args.eval_file, 'w') as eval_file:
        eval_csv = csv.writer(eval_file)
        eval_csv.writerows(eval_set)


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

    # TODO use pydub on eval_set to generate fixed-length evaluation sets

    print("==== Summary ====")
    for input_summary in file_data:
        print("Input counts ({0}): {2}, used: {1}, skipped: {3}".format(*input_summary))
    print("Training set count ({}): {}".format(args.train_file, len(train_set)))
    print("Evaluation set count ({}): {}".format(args.eval_file, len(eval_set)))
