# Add the full path to your audio
PATH_TO_YOUR_AUDIO = 'D:/ZHUANTI/input_audio.wav'

# Load audio with specified sampling rate
import librosa
audio, sr = librosa.load(PATH_TO_YOUR_AUDIO, sr=None)

# Save audio with specified sampling rate
import soundfile as sf
sf.write('input_audio.wav', audio, sr, format='wav')