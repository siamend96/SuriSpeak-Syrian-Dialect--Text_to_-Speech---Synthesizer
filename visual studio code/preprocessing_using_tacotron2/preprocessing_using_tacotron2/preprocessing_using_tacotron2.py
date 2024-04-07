import os 
import sys
import torch
import librosa
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
from scipy.io import wavfile
import soundfile as sf

# Add the directory containing audio_processing.py to the module search path
sys.path.append("/Users/zayde/OneDrive/Desktop/SuriSpeak-Syrian-Dialect--Text_to_-Speech---Synthesizer/tacotron2-master/")
output_dir = "/Users/zayde/OneDrive/Desktop/SuriSpeak-Syrian-Dialect--Text_to_-Speech---Synthesizer/temp/"

# Import functions from the audio_processing module
from audio_processing import window_sumsquare, griffin_lim, dynamic_range_compression, dynamic_range_decompression

# Check if GPU is available
if torch.cuda.is_available():
    # Set the device to GPU
    device = torch.device("cuda")
    print("GPU is available. Using GPU for computation.")
else:
    # If GPU is not available, fallback to CPU
    device = torch.device("cpu")
    print("GPU is not available. Using CPU for computation.")

# Path to the directory containing WAV files
wav_dir = "/Users/zayde/OneDrive/Desktop/SuriSpeak-Syrian-Dialect--Text_to_-Speech---Synthesizer/temp/"

# List of WAV files to process
wav_files = ["ARA NORM  0002.wav"]

# Iterate over each WAV file
for wav_file in wav_files:
    # Load the WAV file
    audio, sr = librosa.load(os.path.join(wav_dir, wav_file), sr=None)
    
    # Convert the audio data to PyTorch tensor and move it to the selected device
    audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device).clone().detach()
    
    # Apply audio processing functions
    # Example usage:
    envelope = window_sumsquare('hann', n_frames=1000, hop_length=256, win_length=1024, n_fft=1024)
    compressed_audio = dynamic_range_compression(audio_tensor, C=0.5)
    
    # Perform other processing tasks as needed
    
    # Save the processed audio
    processed_wav_path = os.path.join(output_dir, "processed_" + wav_file)

    # Write the processed audio data to a WAV file
    sf.write(processed_wav_path, compressed_audio.cpu().numpy(), sr)
