import os
import re
import csv
import argparse
from collections import Counter

import wget
import requests


def make_args():
    parser = argparse.ArgumentParser(description="VoxForge dataset downloader.")
    parser.add_argument('--per-user', default=15, type=int, help="Limit the number of recordings per user")
    parser.add_argument('-d', '--output-dir', default='audiolingua_samples', help="Directory to output wave files to")
    parser.add_argument('-l', '--output-log', default='audiolingua_samples.csv', help="Metadata about downloaded files")
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

        index_url = base_index_url.format(lang=lang_code)
        params = {lang_code: ''}
        page_start = 0
        page_found = True

        user_archives = Counter()

        # Pagination
        while page_found:
            params['debut_articles'] = page_start
            resp = requests.get(index_url, params=params)

            if resp.status_code != 200:
                # The end of pagination
                break
            page_start += 5

            recordings = re.findall(r'\<article.+?(auteur\d+).+?href="([\w\/\-\._]+\.mp3)".+?\</article\>', resp.text, flags=re.DOTALL)
            for user, recording_url in recordings:
                if user_archives[user] >= args.per_user:
                    # We have enough archives of this user
                    continue

                user_archives[user] += 1

                # Download the archive
                recording_name = re.match(r'.+\/([^\/]+)', recording_url).group(1)
                recording_filename = os.path.join(args.output_dir, recording_name)
                download_url = '{}/{}'.format(base_url, recording_url)
                wget.download(download_url, out=recording_filename)
                print()

                log_csv.writerow([recording_name, lang_name, user, user_archives[user]])
                log_file.flush()
        print("Recordings by {} users.".format(len(user_archives)))
    log_file.close()
