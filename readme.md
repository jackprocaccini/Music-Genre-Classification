# Music Genre Classifier
## Jack Procaccini, Dante Vattimo and Matt Rowntree

A Convolutional Neural Network written in Python with the goal of identifying music genres within a given data set.
Code works for [GTZAN](http://marsyas.info/downloads/datasets.html) data set and [Google's](https://research.google.com/audioset/index.html) Audio Set.

Preprocessed data is NOT included - it brings us over github's 100mb limit. To preprocess data:
* For GTZAN data set, download the GTZAN data set (sometimes called marsyas data set in this project, after the website it was downloaded from), which is ~1.2Gb in size and run preprocessor.py
* For Audio Set, run audio_set_download.py, which will download all data from the Google Audio Set via youtube-dl and ffmpeg. Note that this is *slow* and will take up about 1.7Gb in whatever directory you run audio_set_download.py in. Once AudioSet has been downloaded, edit the variables in preprocessor.py (you'll see them marked with comments) and then run preprocessor.py

Once you have preprocessed the data, run cnn.py

## Libraries Used
* [TensorFlow](https://www.tensorflow.org/)
* [Librosa](https://librosa.org/doc/latest/index.html)
    * Librosa's [Github](https://github.com/librosa/librosa)
* [scikit-learn](https://scikit-learn.org/stable/)
* [NumPy](https://numpy.org/)