# Import libraries
import numpy as np
import librosa
from pydub import AudioSegment
import IPython.display as ipd
import matplotlib.pyplot as plt

# Set absolute path of audio files
audio_path_wav = 'C:/Users/Harry/Downloads/sound1.wav'
audio_path_wav2 = 'C:/Users/Harry/Downloads/sound2.wav'

# Get numeric data and sample rate from the audio file
y, sr = librosa.load(audio_path_wav)
y2, sr2 = librosa.load(audio_path_wav2)

# Listen to it
ipd.Audio(data=y, rate=sr)
ipd.Audio(data=y2, rate=sr2)


D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
D2 = librosa.amplitude_to_db(librosa.stft(y2), ref=np.max)

plt.figure(figsize=(12, 8))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

plt.figure(figsize=(56, 8))
librosa.display.specshow(D2, sr=sr2, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

duration = librosa.get_duration(y=y, sr=sr)
time = librosa.times_like(y, sr=sr)

plt.figure(figsize=(15,4))
librosa.display.waveshow(y, sr=sr, x_axis='time')
plt.xlabel("time(s)")
plt.ylabel("amplitude")
plt.title("waveform")
