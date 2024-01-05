import argparse
from pytube import YouTube
from moviepy.editor import VideoFileClip
import os

def download_and_cut_clip(video_url, start_time=0):

    # Init
    yt = YouTube(video_url)
    total_clip_length = 30
    

    # Check if the video is over 30 seconds and clip it accordingly 
    if (yt.length < 30):
        total_clip_length = yt.length
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download YouTube video and create a 30-second clip.')
    parser.add_argument('video_url', type=str, help='YouTube video URL')
    parser.add_argument('start_time', type=int, help='Start time for the clip in seconds')

    args = parser.parse_args()

    # Create 'assets' folder if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    os.makedirs(os.path.join('assets', 'wav'), exist_ok=True)
    os.makedirs(os.path.join('assets', 'mp4'), exist_ok=True)

    download_and_cut_clip(args.video_url, args.start_time)

