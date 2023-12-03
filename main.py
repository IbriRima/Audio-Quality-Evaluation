import os
import json
import requests
import logging
import logging.handlers
import argparse
from bson.binary import Binary

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.cluster import KMeans
import librosa

from pymongo import MongoClient



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_file_handler = logging.handlers.RotatingFileHandler(
    "status.log",
    maxBytes=1024 * 1024,
    backupCount=1,
    encoding="utf8",
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger_file_handler.setFormatter(formatter)
logger.addHandler(logger_file_handler)

try:
    SOME_SECRET = os.environ["SOME_SECRET"]
except KeyError:
    SOME_SECRET = "Token not available!"


mongo_host = 'localhost'
mongo_port = 27017
mongo_db = 'records'


# Create a MongoDB client
client = MongoClient(
    host=mongo_host,
    port=mongo_port,

)



def json_data(json_file_path):
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"The file '{json_file_path}' does not exist.")
    # Load JSON data from the file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def download_and_save(json_file_path):
    data=json_data(json_file_path)
    # Create a DataFrame from the JSON data
    df = pd.DataFrame(data)
    for url in df['record']:
        mp3_record = extract_record_name(url)
        file_path = "./mp3_records/" + mp3_record
        if os.path.exists(file_path):
            print(f"The file {mp3_record} already exists. Skipping download!")
        else:
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"File {mp3_record} downloaded successfully.")
            else:
                print(f"Failed to download  {mp3_record}. Status code {response.status_code}")

def display_spectogram(audio):
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    S_db.shape
    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(S_db,
                                x_axis='time',
                                y_axis='log',
                                ax=ax)
    ax.set_title('Spectogram', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.show()
    return S_db

def extract_record_name(url):
    url_sections= url.split("/")
    record = url_sections[-1]
    return record


def display_audio (folder_path,save_folder='./plots'): 
    os.makedirs(save_folder, exist_ok=True)
    files = os.listdir(folder_path)
    print(files)
    for f in files:
        audio, sample_rate = librosa.load(folder_path+'/'+f, sr=None)
        time = librosa.times_like(audio, sr=sample_rate)
        plt.figure(figsize=(15, 5))
        plt.plot(time, audio, label='Waveform')
        plt.title('Raw Audio')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        record=extract_record_name(f)
        os.makedirs(save_folder+'/waveform', exist_ok=True)
        # Save the waveform plot
        waveform_save_path = os.path.join(save_folder+'/waveform', record+'_waveform.png')
        plt.savefig(waveform_save_path)
        plt.close()
        # Trim and plot the trimmed waveform
        y_trimmed, _ = librosa.effects.trim(audio, top_db=20)

        pd.Series(y_trimmed).plot(lw=1, title='Raw Audio Trimmed')
        os.makedirs(save_folder+'/trimmed', exist_ok=True)
        # Save the trimmed waveform plot
        trimmed_save_path = os.path.join(save_folder+'/trimmed', record+'trimmed_waveform.png')
        
        plt.savefig(trimmed_save_path)
        plt.close()

def display_audio(path):
    ipd.Audio(path)

def get_all_records(directory_path):
    files_and_directories = os.listdir(directory_path)
    records_list=[]
    for item in files_and_directories:
        full_path = os.path.join(directory_path, item)
        if os.path.isfile(full_path):
            records_list.append(item)
    return records_list

def estimate_noise(signal, speech_estimate):
    # Estimate noise by subtracting the estimated speech from the original signal
    noise_estimate = signal - speech_estimate
    return noise_estimate

def calculate_snr(rms_audio, rms_noise):
    snr = 20 * np.log10( rms_audio / rms_noise )
    return snr

def calculate_rms(signal):
    return np.sqrt(np.mean(signal**2))

def calculate_MFCCs(audio,sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=50).T, axis=0)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    return mfcc,mfccs,mfccs_delta,mfccs_delta2

def calculate_autocorrelation(audio_signal):
    autocorr = np.correlate(audio_signal, audio_signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr /= np.max(autocorr)
    return autocorr

def calculate_fundamental_frequency(autocorr, sample_rate):
    peak_index = np.argmax(autocorr[1:]) + 1
    fundamental_freq = sample_rate / peak_index
    return fundamental_freq

def calculate_thd(audio_signal, fundamental_frequency):
    spectrum = np.fft.fft(audio_signal)

    fundamental_index = int(fundamental_frequency * len(audio_signal) / 44100) 
    
    # Calculate THD
    harmonic_indices = [2, 3, 4, 5] 
    thd = np.sqrt(np.sum(np.abs(spectrum[harmonic_indices])**2)) / np.abs(spectrum[fundamental_index]) * 100
    
    return thd

def calculate_dynamic_range_db( audio,rms_audio):
    peak_amplitude = np.max(np.abs(audio))
    dynamic_range_db = 20 * np.log10(peak_amplitude / rms_audio)
    return dynamic_range_db

def calculate_speech_level(audio,sample_rate):
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    speech_level = np.mean(spectral_centroid)
    return speech_level

def audio_features(file_path):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=None)
   
        # Calculate audio RMS
        rms_audio = calculate_rms(audio)

        #Calculate dynamic range of the audio signal in decibels
        dynamic_range_db=calculate_dynamic_range_db( audio,rms_audio)

        # Calculate Speech Level
        speech_level = calculate_speech_level(audio,sample_rate)

        # Pre-emphasis to the audio signal using a pre-emphasis filter
        speech_estimate = librosa.effects.preemphasis(audio)

        # Estimate noise
        noise_estimate = estimate_noise(audio, speech_estimate)

        # Calculate noise RMS
        rms_noise = calculate_rms(noise_estimate)

        # Calculate SNR ratio
        snr=calculate_snr(rms_audio,rms_noise)

        return speech_level, snr,rms_noise,sample_rate,dynamic_range_db,rms_audio
    
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

def create_dataframe(csv_file,mp3_directory='./mp3_records/'):
    data_list=[]
    records_list=get_all_records(mp3_directory)
    for record in records_list:
        info_quality = audio_features('./mp3_records/' + record)
        if info_quality is not None:
            data_list.append({'record': record, 'speech_level': info_quality[0], 'snr': info_quality[1],\
                            'rms_noise':info_quality[2],'sample_rate':info_quality[3],\
                            'dynamic_range_db':info_quality[4],'rms_audio':info_quality[5]})
    data=pd.DataFrame(data_list) 
    os.makedirs('./data', exist_ok=True)
    data.to_csv('./data/'+csv_file,index=False)
    return data

def data_storage(data):

    # Connect to MongoDB
    db = client[mongo_db]
    metadata_collection = db['audios']

    # Read and insert each audio file in the DataFrame with it's metadata
    for index, row in data.iterrows():
        audio_file_path = f"./mp3_records/{row['record']}"
        
        # Transform audio file to binary format
        with open(audio_file_path, 'rb') as audio_file:
            audio_binary = Binary(audio_file.read())
        
        # Create a document with metadata and binary data
        record_document = {
            'filename': row['record'],
            'speech_level': row['speech_level'],
            'snr': row['snr'],
            'rms_noise': row['rms_noise'],
            'sample_rate': row['sample_rate'],
            'dynamic_range_db': row['dynamic_range_db'],
            'audio_data': audio_binary
        }

        # Insert the document into the MongoDB collection
        metadata_collection.insert_one(record_document)

# To be sure that it will be relevant to have 2 classes for our dataset we will be using the elbow method
def elbow_method(data_clustering,max_iter,n_init):
    num_clusters_range = range(1, 11)  # Try clusters from 1 to 10
    sse = [] 
    for num_clusters in num_clusters_range:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42,n_init=n_init,max_iter=max_iter)
        kmeans.fit(data_clustering)
        sse.append(kmeans.inertia_)

    plt.plot(num_clusters_range, sse, marker='o')
    plt.title('Elbow Plot')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Distances')

    # Display the plot
    plt.show()

    # Save the plot
    plt.savefig('./plots/elbow_plot.png')


def clustering(data,max_iter,n_init,n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init=n_init,max_iter=max_iter)
    kmeans.fit(data)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    data['label']=labels
    plt.scatter(data_cluster.iloc[:, 0], data_cluster.iloc[:, 1], c=labels, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=20, label='Cluster Centers')
    plt.title('K-Means Clustering')
    plt.xlabel('rms_noise')
    plt.ylabel('speech_level')
    plt.legend()

    # Save the plot
    plt.savefig("./plots/kmeans_clustering_plot.png", dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

    return data
    


def labels_pie_plot(data_cluster):
    # Count the occurrences of each unique value in the 'label' column
    label_counts = data_cluster['label'].value_counts()

    # Create a Pie chart using Plotly
    fig = px.pie(
        label_counts,
        values=label_counts,
        names=label_counts.index,
        title='Pie Plot for Cluster Labels',
        labels={'label': 'Cluster Labels'},
        template='plotly',
    )

    # Save the plot as an a png image
    fig.write_image("./plots/Labels_pie_plot.png", format='png', scale=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["mp3", "audio_viz", "save_data", "elbow_method", "audios_classification"],
                        help="Choose action: mp3, audio_viz, save_data, elbow_method, or audios_classification")

    # Add optional parameters for audios_classification
    parser.add_argument("--features", nargs="+", help="Provide the list of features for clustering")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # paths
    json_file_path = "./speechcrawling/records.json"
    mp3_folder = './mp3_records'
    viz_folder='./plots'
 
    args =parse_args()

    if args.action == "mp3":
        logger.info(f"Downloading mp3 files from {json_file_path}")

    elif args.action == "audio_visualization":
        logger.info("Performing audio visualizations")
        display_audio (mp3_folder,save_folder='./plots')


    elif args.action== "save_data":
        logger.info("Inserting records into MongoDB database")
        data=create_dataframe('./records.csv','./mp3_records/')
        data=pd.read_csv('./data/records.csv')
        data_storage(data)

    elif args.action== "elbow_method":
        logger.info("Inserting records into MongoDB database")
        data=pd.read_csv('./data/records.csv')
        data_clustering=data[['rms_noise','speech_level','dynamic_range_db']]
        elbow_method(data_clustering)

    elif args.action == "audios_classification" and args.features:
        logger.info("Classifying audios based on their quality")
        features_to_use = args.features
        data = pd.read_csv('./data/records.csv')

        # Check if specified features exist in the dataset
        invalid_features = [feature for feature in features_to_use if feature not in data.columns]
        if invalid_features:
            logger.warning(f"Invalid feature(s) specified: {', '.join(invalid_features)}")
        else:
            data_cluster = data[features_to_use]
            data_labeled = clustering(data_cluster, max_iter=15, n_init='auto', n_clusters=2)
            data_labeled.to_csv('Classified_audio.csv', index=False)


    else:
        logger.warning("Invalid action provided.")

        