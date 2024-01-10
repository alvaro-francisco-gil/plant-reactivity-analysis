from scipy.io import wavfile
import matplotlib.pyplot as plt
import os

def plot_waveform_with_range_in_seconds(wav_file, start_time, end_time):
    # Read the WAV file
    sample_rate, data = wavfile.read(wav_file)

    # Calculate the number of samples for the given time range
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Extract the waveform data within the specified time range
    waveform_range = data[start_sample:end_sample]

    # Plot the waveform
    plt.figure(figsize=(10, 6))
    plt.plot(waveform_range)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform between {start_time} and {end_time} seconds')
    plt.show()

def plot_waveform(data):

    # Plot the entire waveform
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()

def plot_wav_file(file):

    print(file)
    sample_rate, data = wavfile.read(file)
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()

def find_wav_files(directory):
    wav_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                wav_files.append(file_path)

    return wav_files

def execute_function_in_subfolders(folder_path, function):
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            subfolder_path = os.path.join(root, dir)
            function(subfolder_path)

def find_wav_files(directory):
    wav_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                wav_files.append(file_path)

    return wav_files