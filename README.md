# 📢 Audio Denoising Pipeline (DCCRNet + DSP + VAD)

This repository implements a comprehensive audio denoising pipeline using a pretrained deep learning model (DCCRNet) from the Asteroid library, combined with traditional audio engineering techniques and voice activity detection (VAD). The pipeline is optimized for enhancing noisy `.wav` files in English, especially in benchmark speech corpora.

---

## 📁 Folder Structure

```
.
├── noise-reduction-script.py   # Main processing script
├── ms-benchmarking-8/wav/      # Input folder with .wav files
└── output: <input>-cleaned.wav # Output will be saved with `-cleaned` suffix
```

---

## ⚙️ Dependencies

Install all required libraries via pip:

```bash
pip install torch torchaudio librosa soundfile scipy webrtcvad tqdm asteroid
```

---

## 🧠 Pipeline Overview

### 1. **Deep Learning Enhancement (DCCRNet)**
- **Model**: `JorisCos/DCCRNet_Libri1Mix_enhsingle_16k` from HuggingFace.
- **Role**: Denoises the waveform in overlapping chunks using a pre-trained neural network trained on Libri1Mix.
- **Blending**: Output is a soft blend of the original and enhanced signal (`0.7 * original + 0.3 * enhanced`).

### 2. **Noise Profiling**
- Computes RMS and spectral statistics from the first 500 ms of audio to inform adaptive filtering.

### 3. **Traditional DSP Filters**
- **WebRTC VAD**: Removes unvoiced (silent or noisy) regions.
- **Adaptive Bandpass Filter**: Automatically sets cutoff frequencies based on spectral centroid.
- **Low Shelf Filter**: Boosts lower frequencies (cutoff: 200 Hz, gain: +1 dB).
- **High Shelf Filter**: Boosts higher frequencies (cutoff: 4 kHz, gain: +1.5 dB).
- **Spectral Gate (Optional)**: Suppresses background noise in frequency bins below a certain threshold.

### 4. **Peak Normalization**
- Final step to ensure output audio is amplitude-normalized without clipping.

---

## 🧪 How It Works

### Run the script:

```bash
python noise-reduction-script.py
```

### Default behavior:
- All `.wav` files in `ms-benchmarking-8/wav/` will be denoised.
- Cleaned files are saved in the same folder with a `-cleaned.wav` suffix.

---

## 🔍 Key Functions

| Function | Purpose |
|---------|---------|
| `run_denoise_chunks()` | Applies DCCRNet on audio chunks to handle long files without memory overflow. |
| `get_noise_stats()` | Extracts spectral stats from audio head for adaptive filter design. |
| `apply_bandpass()` | Applies a standard bandpass filter. |
| `apply_adaptive_bandpass()` | Adapts bandpass range based on spectral centroid. |
| `apply_low_shelf()` / `apply_high_shelf()` | Emulates EQ adjustments to improve clarity. |
| `apply_webrtc_vad()` | Removes silence/noise using WebRTC VAD. |
| `apply_spectral_gate()` | Optional: noise gating via STFT. |
| `normalize_peak()` | Rescales audio to prevent clipping. |
| `process_audio_folder()` | Full batch processor for all `.wav` files. |

---

## 🧱 Customization

- **Input Folder**: Change `'ms-benchmarking-8/wav'` to your target directory.
- **Chunk Size**: Modify `chunk_seconds=10` in `run_denoise_chunks()` if dealing with longer/shorter audio files.
- **VAD Sensitivity**: Adjust `level=2` (0=aggressive, 3=permissive).
- **EQ Gains**: Tune `gain_db` in `apply_low_shelf` and `apply_high_shelf`.
- **Enable Gating**: Uncomment `apply_spectral_gate()` in `process_audio_folder()` if background noise persists.

---

## ❗ Notes

- Model is optimized for English and might underperform on other languages.
- Ensure input audio is 16kHz for compatibility.
- The script is designed for batch processing and robustness but may fail on extremely short or corrupted files.

---

## 🧑‍🔧 Author

Shawn Yap Zheng Yi  
Under the supervision of Rachel LW Tan  
xData, Home Team Science and Technology Agency (HTX)
