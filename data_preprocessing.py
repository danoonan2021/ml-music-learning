import os
import json
import librosa
import math

# init
path_to_dataset = './gtzan/genres_original'
WAV_DURATION = 30
DEFAULT_SR = 22050
path_to_json = 'vocab.json'
SAMPLES_PER_TRACK = DEFAULT_SR * WAV_DURATION

# converts a waveform audio file to an mfcc and dumps the data into json file
def convert_to_mfcc(dataset_path, json_path, num_mfcc=13, num_fft=2048, hop_length=512, num_segments=5):
    # get the # of samples per segment
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    # get the expected # of mfcc vectors per segment
    expected_number_of_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # json format
    data = {
        'mapping': [],
        'mfcc': [],
        'labels': []
    }
    
    # iterating through dataset
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path_to_dataset)):

        # iterating through subdirectories i.e. genres
        if dirpath is not path_to_dataset:

            genre_name = os.path.basename(dirpath)
            data['mapping'].append(genre_name)

            # iterate through each waveform audio file and load it for librosa 
            for f in filenames:
                try: 
                    file_path = os.path.join(dirpath, f)
                    print(file_path)

                    y, sr = librosa.load(file_path)

                    # iterate through segments of the audio file
                    for s in range(num_segments):
                        start = samples_per_segment * s
                        end = samples_per_segment + start

                        # convert the segment into an mfcc
                        mfcc = librosa.feature.mfcc(y=y[start:end],
                                                    sr=sr,
                                                    n_fft=num_fft,
                                                    n_mfcc=num_mfcc,
                                                    hop_length=hop_length)
                        # Transpose mfcc and add it to json map
                        mfcc = mfcc.T


                        if len(mfcc) == expected_number_of_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                except Exception as e:
                    print("File" + f + "is corrupt")
                    continue
    
    # dump the data into the json object (created if not already existing)
    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)

# function call
convert_to_mfcc(path_to_dataset, path_to_json, num_segments=10)
