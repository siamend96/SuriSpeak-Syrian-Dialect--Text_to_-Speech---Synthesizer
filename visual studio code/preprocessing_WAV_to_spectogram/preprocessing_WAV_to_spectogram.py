import os
import torch
import torchaudio
import matplotlib.pyplot as plt

# Input and output directory
input_dir = '/Users/zayde/OneDrive/Desktop/SuriSpeak Syrian Dialect  Text_to_ Speech   Synthesizer/temp/preprocessing'
output_dir = '/Users/zayde/OneDrive/Desktop/SuriSpeak Syrian Dialect  Text_to_ Speech   Synthesizer/datasets/arabic-speech-corpus/spectogram'

def create_spectrogram(input_dir, output_dir, sr=44100, n_fft=2048, hop_length=512):
    """
    Convert WAV files into spectrogram images using PyTorch.

    Args:
    - input_dir (str): Path to the directory containing input WAV files.
    - output_dir (str): Path to the directory where spectrogram images will be saved.
    - sr (int): Sample rate for audio processing (default: 44100).
    - n_fft (int): Size of FFT window (default: 2048).
    - hop_length (int): Number of samples between successive frames (default: 512).
    """
    # Check if GPU is available
    if torch.cuda.is_available():
        print("GPU found. Using GPU for computation.")
    else:
        print("GPU not found. Using CPU for computation.")
        return 1

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Move computation to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Iterate over WAV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            try:
                # Load audio file
                audio_path = os.path.join(input_dir, filename)
                waveform, _ = torchaudio.load(audio_path)

                # Move data to GPU if available
                waveform = waveform.to(device)

                # Compute spectrogram
                spectrogram = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr, n_fft=n_fft, hop_length=hop_length).to(device)(waveform)

                # Convert to decibels (log scale)
                spectrogram_db = torchaudio.transforms.AmplitudeToDB().to(device)(spectrogram)

                # Move spectrogram back to CPU for plotting
                spectrogram_db = spectrogram_db.cpu()

                # Plot and save spectrogram
                plt.figure(figsize=(10, 6))
                plt.imshow(spectrogram_db[0].numpy(), cmap='magma', origin='lower')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel Spectrogram')
                plt.xlabel('Time')
                plt.ylabel('Frequency')
                plt.tight_layout()
                output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_spectrogram.png')
                plt.savefig(output_path)
                plt.close()
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
create_spectrogram(input_dir, output_dir)


