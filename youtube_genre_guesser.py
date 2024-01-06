import argparse
from pytube import YouTube
from moviepy.editor import VideoFileClip
import os
import librosa
import numpy as np
import tensorflow as tf
import math

GENRES = ['Blues', 'Classical', 'Country', 'Disco', 'Hip Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

def download_and_cut_clip(video_url, start_time=0):

    # Init
    yt = YouTube(video_url)
    total_clip_length = 30
    video_len = yt.length
    
    # Check if the provided start time cannot provide a 30 second clip
    assert (start_time + 30) < video_len, "Warning: Provided Start Time does not allow for a 30 second clip, no prediction made."
    
    # Check if the video is over 30 seconds and clip it accordingly 
    if (video_len < 30):
        total_clip_length = video_len
    else: 
        total_clip_length = start_time + 30
    
    # Download YouTube video
    video = yt.streams.get_highest_resolution()
    video.download('./assets/mp4')
    video_path = os.path.join('assets', 'mp4', video.default_filename)

    video_filename = video.default_filename  # Get the downloaded video filename

   # Extract audio from the video and save it as WAV
    audio_clip = VideoFileClip(video_path).subclip(start_time, total_clip_length).audio
    audio_filename = f"{os.path.splitext(video_filename.replace(' ', '_'))[0]}_{start_time}s_audio.wav"
    audio_path = os.path.join('assets', 'wav', audio_filename)
    audio_clip.write_audiofile(audio_path)
    print(f"30-second clip '{audio_filename}' created successfully!")

    # Close the resources
    audio_clip.close()
    del audio_clip

    return audio_path

# Have the model predict the genre of the youtube clip
def model_predict_genre(path_to_clip, model_name):
    data = []
    # Load the .wav file
    y, sr = librosa.load(path_to_clip)
    print(len(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=2048, n_mfcc=13, hop_length=512)
    mfcc = mfcc.T
        
    # Calculate the number of frames and features of your MFCC
    num_frames = mfcc.shape[1]
    num_features = mfcc.shape[0]

    # Define the desired number of frames and features
    desired_frames = 130
    desired_features = 13

    # Pad or slice the frames and features accordingly
    if num_frames > desired_frames:
        mfcc = mfcc[:, :desired_frames]
    elif num_frames < desired_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, desired_frames - num_frames)), mode='constant')

    if num_features > desired_features:
        mfcc = mfcc[:desired_features, :]
    elif num_features < desired_features:
        mfcc = np.pad(mfcc, ((0, desired_features - num_features), (0, 0)), mode='constant')

    # Reshape the MFCC array to match the expected input shape (None, 130, 13, 1)
    input = mfcc.T[np.newaxis, ..., np.newaxis]
    
    # Check if the user inputted the file ext.
    if len(model_name) != 19:
        model_name = f"{model_name}.py"

    # Load the model
    loaded_model = tf.keras.models.load_model(model_name)

    # Pass input to model
    print("----- Starting model prediction -----")
    prediction = loaded_model.predict(input)

    # Get the genre
    predicted_index = np.argmax(prediction, axis=1)

    for i in range(len(GENRES)):
        if predicted_index[0] == i:
            predicted_genre = GENRES[i]
    
    print(f"Predicted Genre: {predicted_genre}")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download YouTube video and create a 30-second clip.')
    parser.add_argument('video_url', type=str, help='YouTube video URL')
    parser.add_argument('start_time', type=int, help='Start time for the clip in seconds')
    parser.add_argument('model_name', type=str, help='Give the name of the model that you want to guess')

    args = parser.parse_args()

    # Create 'assets' folder if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    os.makedirs(os.path.join('assets', 'wav'), exist_ok=True)
    os.makedirs(os.path.join('assets', 'mp4'), exist_ok=True)

    # Download the youtube video and clip it into a .wav file
    path_to_clip = download_and_cut_clip(args.video_url, args.start_time)

    # Predict the genre of the clip
    model_predict_genre(path_to_clip, args.model_name)

