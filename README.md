# Spoken Language Identification from Short Utterances

This is a model for identifying the language spoken in a short audio segment.

## Installation

To install the required libraries (tested on Ubuntu 17.11) run:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Predicting the language in an audio file

1. Convert an audio file to a spectrogram:

        python data/dataset_gen.py -z speech.wav -o .

1. Obtain the prediction using a pre-trained model:

        python main.py --model-dir your-trained-model/ --params your-trained-model/params.json --model combo --predict speech.png


## Training the model from scratch

1. Prepare a dataset:
   - Place your spectrograms in a folder
   - Create a test set CSV file containing "Filename,Language" pairs
   - Create an evaluation set CSV file (same format as the test)

1. Train the model:

        python main.py --model-dir your-trained-model/ --params your-trained-model/params.json --model combo --image-dir your-data/ --train-set your-data/train-set.csv --eval-set your-data/eval-set.csv

# Author

This project was developed by [Rimvydas Naktinis](https://github.com/naktinis) during [Pi School's AI programme](http://picampus-school.com/programme/school-of-ai/) in Fall 2017.

![photo of Rimvydas Naktinis](http://picampus-school.com/wp-content/uploads/2017/11/IMG_2135-150x150.jpg)
