import os
import re
import csv
import argparse
from collections import Counter
from collections import OrderedDict

import wget
import requests
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


def make_args():
    parser = argparse.ArgumentParser(description="VoxForge dataset downloader.")
    parser.add_argument('--per-user', default=15, type=int, help="Limit the number of recordings per user")
    parser.add_argument('-d', '--output-dir', default='audiolingua_samples', help="Directory to output wave files to")
    parser.add_argument('-l', '--output-log', default='audiolingua_samples.csv', help="Metadata about downloaded files")
    parser.add_argument('-s', '--split', default=20, type=int, help="Split each file into tracks of the given length (in seconds)")
    return parser.parse_args()


if __name__ == '__main__':
    base_url = 'https://www.audio-lingua.eu'
    base_index_url = '{}/spip.php'.format(base_url)
    languages = {
        'Italian': 'rubrique6',
        'French': 'rubrique1',
        'Portuguese': 'rubrique8',
        'German': 'rubrique3',
        'English': 'rubrique2',
        'Spanish': 'rubrique4',
    }

    args = make_args()
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = open(args.output_log, 'w')
    log_csv = csv.writer(log_file, lineterminator='\n')
    for lang_name, lang_code in languages.items():
        print("Downloading archives for {}.".format(lang_name))

        max_per_lang = 1500
        index_url = base_index_url.format(lang=lang_code)
        params = OrderedDict({lang_code: ''})
        page_start = 0
        page_found = True

        user_recordings = Counter()

        # Pagination
        while page_found:
            params['debut_articles'] = page_start
            resp = requests.get(index_url, params=params)

            if resp.status_code != 200 or page_start > max_per_lang:
                # The end of pagination
                break
            page_start += 5

            # Extract item count
            max_per_lang = int(re.search(r'(\d+) ressources', resp.text).group(1))

            recordings = re.findall(r'\<article.+?(auteur\d+).+?href="([\w\/\-\._]+\.mp3)".+?\</article\>', resp.text, flags=re.DOTALL)
            for user, recording_url in recordings:
                if user_recordings[user] >= args.per_user:
                    # We have enough archives of this user
                    continue

                # Download the archive
                recording_name = re.match(r'.+\/([^\/]+)', recording_url).group(1)
                recording_filename = os.path.join(args.output_dir, recording_name)
                download_url = '{}/{}'.format(base_url, recording_url)
                wget.download(download_url, out=recording_filename)
                print()

                if args.split > 0:
                    try:
                        recording = AudioSegment.from_mp3(recording_filename)
                    except CouldntDecodeError:
                        print("Error decoding {}".format(recording_filename))
                        continue

                    recording_slices = recording[::args.split * 1000]
                    for slice_num, rec_slice in enumerate(recording_slices):
                        # Make the slice of filename.mp3 look like filename_1.mp3
                        slice_name = '{0}_{2}.mp3'.format(*os.path.splitext(recording_name), slice_num + 1)
                        slice_path = os.path.join(args.output_dir, slice_name)
                        print(slice_path)

                        if len(rec_slice) > 3000:
                            # Only save slices longer than 3 seconds
                            user_recordings[user] += 1
                            rec_slice.export(slice_path, format='mp3')
                            log_csv.writerow([slice_name, lang_name, user, user_recordings[user]])
                    os.remove(recording_filename)
                else:
                    user_recordings[user] += 1
                    log_csv.writerow([recording_name, lang_name, user, user_recordings[user]])
                log_file.flush()
        print("Recordings by {} users.".format(len(user_recordings)))
    log_file.close()
