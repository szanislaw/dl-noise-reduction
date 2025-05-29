import os
import gradio as gr
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile

# === PATHS ===
original_folder = "airline-radio-comms"
enhanced_folder = original_folder + "-asteroid-vad-enhanced"

def plot_waveform_and_mel(audio_path, title_prefix, sr_target=16000, vmin=-80, vmax=0):
    y, sr = librosa.load(audio_path, sr=sr_target)

    # === Compute mel-spectrogram ===
    mel = librosa.feature.melspectrogram(y=y, sr=sr_target, n_fft=2048, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    duration = librosa.get_duration(y=y, sr=sr_target)

    # === Plot both waveform and mel ===
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot waveform
    librosa.display.waveshow(y, sr=sr_target, ax=axs[0])
    axs[0].set_title(f"{title_prefix} - Waveform")
    axs[0].set_ylabel("Amplitude")
    axs[0].label_outer()

    # Plot mel-spectrogram with fixed color scale and frequency axis
    img = librosa.display.specshow(
        mel_db, sr=sr_target, hop_length=512, x_axis='time', y_axis='mel',
        ax=axs[1], cmap='magma', vmin=vmin, vmax=vmax
    )
    axs[1].set_title(f"{title_prefix} - Mel Spectrogram")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].label_outer()
    fig.colorbar(img, ax=axs[1], format='%+2.0f dB')

    # Set consistent x-axis limit
    axs[1].set_xlim(0, duration)

    # Save the plot
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.tight_layout()
    plt.savefig(tmpfile.name)
    plt.close(fig)

    return tmpfile.name


def preview_audio(filename):
    original_path = os.path.join(original_folder, filename)
    enhanced_path = os.path.join(enhanced_folder, filename)
    transcript_path = os.path.join("groundtruth", filename.replace(".wav", ".txt"))

    if not os.path.exists(original_path):
        return None, None, None, None, "", f"Original file not found: {filename}"
    if not os.path.exists(enhanced_path):
        return original_path, None, None, None, "", f"Enhanced version not found: {filename}"

    # Generate plots
    orig_plot = plot_waveform_and_mel(original_path, "Original")
    enh_plot = plot_waveform_and_mel(enhanced_path, "Enhanced")

    # Load transcript
    transcript = "Transcript not found."
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()

    return original_path, enhanced_path, orig_plot, enh_plot, transcript, "Preview loaded successfully."


# Get list of .wav files from original folder
file_choices = [f for f in os.listdir(original_folder) if f.lower().endswith('.wav')]

with gr.Blocks() as demo:
    gr.Markdown("### ATC Audio A/B Preview: Original vs Enhanced")
    with gr.Row():
        file_dropdown = gr.Dropdown(choices=file_choices, label="Select File")
        load_button = gr.Button("Load Preview")

    with gr.Row():
        original_audio = gr.Audio(label="🎙️ Original Audio", type="filepath")
        enhanced_audio = gr.Audio(label="🧼 Enhanced Audio", type="filepath")

    with gr.Row():
        original_plot = gr.Image(label="Original Waveform & Mel-Spectrogram")
        enhanced_plot = gr.Image(label="Enhanced Waveform & Mel-Spectrogram")

    transcript_box = gr.Textbox(label="Ground Truth", lines=4, interactive=False)

    status_text = gr.Textbox(label="Status", interactive=False)

    load_button.click(
        fn=preview_audio,
        inputs=file_dropdown,
        outputs=[original_audio, enhanced_audio, original_plot, enhanced_plot, transcript_box, status_text]
    )

demo.launch()

