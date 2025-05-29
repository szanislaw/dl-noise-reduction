import os
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import scipy.signal as signal
import webrtcvad
from asteroid.models import DCCRNet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Load the pre-trained DCCRNet model
# Ensure you have the correct model name
# You can find available models at: https://huggingface.co/models?search=asteroid
# For example, using the DCCRNet model for single-channel speech enhancement
# from the Libri1Mix dataset
# Make sure to install the required libraries:
# pip install torch torchaudio librosa soundfile scipy webrtcvad tqdm

# Model only works well with English 
model = DCCRNet.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
model.to(device)

def clamp_cutoff(cutoff, sr, margin=0.95):
    return min(cutoff, 0.5 * sr * margin)

def apply_bandpass(audio, sr, lowcut=80, highcut=3400, order=4):
    nyquist = 0.5 * sr
    low = clamp_cutoff(lowcut, sr) / nyquist
    high = clamp_cutoff(highcut, sr) / nyquist
    if low >= high or high >= 1.0:
        return audio
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return signal.sosfilt(sos, audio)

def apply_low_shelf(audio, sr, cutoff=200, gain_db=1):
    cutoff = clamp_cutoff(cutoff, sr)
    if cutoff <= 0:
        return audio
    gain = 10**(gain_db / 20)
    b, a = signal.butter(1, cutoff / (0.5 * sr), btype='low')
    return signal.lfilter(b, a, audio) * gain

def apply_high_shelf(audio, sr, cutoff=4000, gain_db=1.5):
    cutoff = clamp_cutoff(cutoff, sr)
    if cutoff >= 0.5 * sr:
        return audio
    gain = 10**(gain_db / 20)
    b, a = signal.butter(1, cutoff / (0.5 * sr), btype='high')
    return signal.lfilter(b, a, audio) * gain

def normalize_peak(audio):
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio

def apply_spectral_gate(audio, sr, threshold_db=-30):
    stft = librosa.stft(audio, n_fft=1024, hop_length=256)
    magnitude, phase = np.abs(stft), np.angle(stft)
    db_mag = librosa.amplitude_to_db(magnitude)
    gate_mask = db_mag > threshold_db
    gated_mag = magnitude * gate_mask
    stft_gated = gated_mag * np.exp(1j * phase)
    return librosa.istft(stft_gated, hop_length=256)

def apply_webrtc_vad(audio, sr, level=2):
    vad = webrtcvad.Vad(level)
    window_ms = 30
    samples_per_window = int(sr * window_ms / 1000)
    bytes_per_sample = 2

    int16_audio = np.int16(audio * 32768)
    pcm = int16_audio.tobytes()

    voiced = bytearray()
    for i in range(0, len(pcm), samples_per_window * bytes_per_sample):
        frame = pcm[i:i + samples_per_window * bytes_per_sample]
        if len(frame) < samples_per_window * bytes_per_sample:
            break
        if vad.is_speech(frame, sample_rate=sr):
            voiced.extend(frame)

    if not voiced:
        return audio
    return np.frombuffer(voiced, dtype=np.int16).astype(np.float32) / 32768.0

def get_noise_stats(audio, sr, window_ms=500):
    n = int((window_ms / 1000) * sr)
    head = audio[:n]
    rms = np.sqrt(np.mean(head**2))
    spec = np.mean(np.abs(librosa.stft(head, n_fft=1024)), axis=1)
    return rms, spec

def apply_adaptive_bandpass(audio, sr, spectrum, margin=0.5):
    centroid = librosa.feature.spectral_centroid(S=np.expand_dims(spectrum, axis=1), sr=sr)[0, 0]
    low = max(50, centroid * (1 - margin))
    high = min(8000, centroid * (1 + margin))
    return apply_bandpass(audio, sr, lowcut=low, highcut=high)

def run_denoise_chunks(input_path, tmp_path, sr_target=16000, chunk_seconds=10):
    waveform, sr = torchaudio.load(input_path)
    if sr != sr_target:
        waveform = torchaudio.transforms.Resample(sr, sr_target)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)

    chunk_len = sr_target * chunk_seconds
    total_len = waveform.shape[-1]
    output = []

    with torch.no_grad():
        for start in range(0, total_len, chunk_len):
            end = min(start + chunk_len, total_len)
            chunk = waveform[:, start:end].to(device)
            try:
                result = model.separate(chunk)[0].squeeze(0).cpu().numpy()
                original = chunk.squeeze(0).cpu().numpy()
                mixed = 0.7 * original + 0.3 * result
                output.append(mixed)
            except RuntimeError as e:
                print(f"Error chunk {start}-{end}: {e}")

    full = np.concatenate(output, axis=-1)
    sf.write(tmp_path, full, sr_target)
    return tmp_path

def process_audio_folder(input_dir):
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    for fname in tqdm(wav_files, desc="Processing files"):
        in_path = os.path.join(input_dir, fname)
        tmp_path = "tmp_asteroid_clean.wav"

        try:
            denoised_path = run_denoise_chunks(in_path, tmp_path)
            audio, sr = librosa.load(denoised_path, sr=None)

            noise_rms, noise_spec = get_noise_stats(audio, sr)
            audio = apply_webrtc_vad(audio, sr, level=2)
            audio = apply_adaptive_bandpass(audio, sr, noise_spec, margin=0.5)
            audio = apply_low_shelf(audio, sr, gain_db=1)
            audio = apply_high_shelf(audio, sr, gain_db=1.5)
            # audio = apply_spectral_gate(audio, sr, threshold_db=-30)  # optionally enable
            audio = normalize_peak(audio)

            base, ext = os.path.splitext(fname)
            out_name = f"{base}-cleaned{ext}"
            out_path = os.path.join(input_dir, out_name)
            sf.write(out_path, audio, sr)
        except Exception as e:
            print(f"Error {fname}: {e}")


# Input your audo folder here
process_audio_folder('ms-benchmarking-8/wav')
