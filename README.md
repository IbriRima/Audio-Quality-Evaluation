# Speech Data Crawler and Classification

This Python program is designed to crawl speech data from the [Freesound](https://freesound.org/charts/) website, save them in a MongoDB database, and subsequently classify all speech audio files by speech quality using Kmeans clustering.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Database](#database)
- [Application features](#App_features)


## Introduction

This project aims to automate the process of crawling speech data, storing it, and classifying audio files based on speech quality metrics.

## Requirements
```
pip -r requirements.txt

```
## Database
MongoDB has robust support for storing binary data, which makes it suitable for storing audio files. Using the BSON data type "Binary" I was able to store audio file content directly within MongoDB next to the audio metadata.
- filename: record name
- speech_level: The speech level is calculated based on the spectral centroid of an audio signal
- rms_audio: Root Mean Square of Audio and it measure the average power of a varying signal. It is a more stable indicator than the peak amplitude, as it considers the entire waveform over a period of time.
- rms_noise: Root Mean Square of Noise. The function estimate_noise takes the original audio signal and subtracts the estimated speech component from it to obtain the noise. The rms_noise is then calculated for this noise signal using the RMS formula.
- snr: Signal-to-Noise Ratio is a measure used to quantify the level of a desired signal relative to the level of background noise.,
- sample_rate: sample rate refers to the number of samples of audio carried per second. A higher sample rate means more samples are taken per second, providing a more accurate representation of the original analog signal. 
- dynamic_range_db: a measure of the span between the quietest and loudest parts of an audio signal,
- audio_data: audio in binary format
  
## App_features
- Download mp3 files: You should start by downloading mp3 files from the crawled urls
```
python main.py mp3

``` 
- Audio visualizations: To be able to get better visualization of the audios you should use
```
python main.py audio_visualization
```
This enable visualization of the raw audio waveform and its trimmed version and saves each of them in the folder plots
- Elbow method: To be sure that our data can really be clustered into two group so we can use this for classification into good or bad quality videos we will be usinf elbow method to visualize optimal clusters number and save the plot in the plots folder. You can use this by running: 
```
python main.py elbow_method

``` 

- Data Saving: This will enable the saving of the audio and its metadata in the MongoDB database. Yu can use this through:
```
python main.py save_data

```
- Audio classification based on their quality: Through the clustering we were able to label the dataset to good quality audios or bad quality audios and to do this you should use:

```
 python main.py audios_classification --features rms_noise speech

```
Keep in mind that after --features you can choose what features to use for the clustering.
The final data with labels will be saved under the data folder as classified_audio_.csv
