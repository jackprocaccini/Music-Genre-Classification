# Music Genre Classifier
## Jack Procaccini, Dante Vattimo and Matt Rowntree

A Convolutional Neural Network written in Python with the goal of identifying music genres within a given data set.
Code works for [GTZAN](http://marsyas.info/downloads/datasets.html) data set and [Google's](https://research.google.com/audioset/index.html) Audio Set.
Preprocessed data is included. If you want to process data on your own, you need to do one of these two things:
* Download the GTZAN data set (sometimes called marsyas data set in this project, after the website it was downloaded from) and run preprocessor.py
* Run audio_set_download.py, which will download all data from the Google Audio Set that is present in the script's initial dictionary. To add more genres than what is included, add the name of the genre (in English, e.g. "blues", "punk", etc.) to the dictionary located in audio_set_download.py, find that genre's id and all of the genre's child ids in ontology.json, and add them to the dictionary. If the genre is not present in the ontology, you cannot use it. The genre name in English is the key and a list of tags are the values for that key. This takes a long time to downlaod; settings that are set by default take about 30 minutes to download and takes up approximately 1.5 gigabytes of space.

## Libraries Used
* [TensorFlow](https://www.tensorflow.org/)
* [Librosa](https://librosa.org/doc/latest/index.html)
    * Librosa's [Github](https://github.com/librosa/librosa)
* [scikit-learn](https://scikit-learn.org/stable/)
* [NumPy](https://numpy.org/)