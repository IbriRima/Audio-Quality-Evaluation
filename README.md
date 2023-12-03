# Speech Data Crawler and Classification

This Python program is designed to crawl speech data from the [Freesound](https://freesound.org/charts/) website, save them in a MongoDB database, and subsequently classify all speech audio files by speech quality using Kmeans clustering.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Database](#database)


## Introduction

This project aims to automate the process of crawling speech data, storing it, and classifying audio files based on speech quality metrics.

## Requirements
```
pip -r requirements.txt

```
## Database
MongoDB has robust support for storing binary data, which makes it suitable for storing audio files. Using the BSON data type "Binary" I was able to store audio file content directly within MongoDB next to the audio metadata.

## App features
- Download mp3 files: You should start by downloading mp3 files from the crawled urls
```
python main.py mp3

``` 
- Audio visualizations: To be able to get better visualization of the audios you should use
```
python main.py audio_visualization
``` 
- [Database](#database)


## Download mp3 files

