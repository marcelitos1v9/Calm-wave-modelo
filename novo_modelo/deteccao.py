import numpy as np
import soundfile as sf
import librosa
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import pygame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

def plot_spectrogram(audio, sr, title, ax):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    ax.clear()
    img = ax.imshow(D, aspect='auto', origin='lower', 
                    extent=[0, len(audio)/sr, 0, sr/2])
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')

def detectar_latido(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    energia = librosa.feature.rms(y=audio)[0]
    limiar = 0.01
    latidos_presentes = energia > limiar
    return audio, sr, np.any(latidos_presentes)

def separar_latidos(audio_file, output_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    energia = librosa.feature.rms(y=audio)[0]
    limiar = 0.01
    latidos_presentes = energia > limiar
    latido_audio = np.zeros_like(audio)

    for i, presente in enumerate(latidos_presentes):
        if presente:
            start = int(max(0, librosa.frames_to_samples(i)))
            end = int(min(len(audio), librosa.frames_to_samples(i + 1)))
            latido_audio[start:end] = audio[start:end]

    # Criação do diretório, se necessário
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_file, latido_audio, sr)

    return latido_audio

def play_audio(audio_file):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
    except Exception as e:
        messagebox.showerror("Error playing audio", str(e))

def process_audio():
    global audio_file, output_file
    audio_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if not audio_file:
        return

    audio_file_path = Path(audio_file)

    print(f"Selected file: {audio_file_path}") # Debug print

    if not audio_file_path.is_file():
        messagebox.showerror("Error", f"File not found: {audio_file_path}")
        return

    try:
        audio, sr, latido_detectado = detectar_latido(audio_file)

        # Construct output path in the same directory as the input file
        output_file = audio_file_path.parent / "latido_separado.wav"

        print(f"Output file path: {output_file}") # Debug print

        latido_audio = separar_latidos(audio_file, str(output_file)) # Convert to string

        if not output_file.is_file():
            messagebox.showerror("Error", f"Failed to create output file: {output_file}")
            return

        plot_spectrogram(audio, sr, "Original Audio Spectrogram", ax_original)
        plot_spectrogram(latido_audio, sr, "Separated Audio Spectrogram", ax_separado)
        canvas.draw()

        btn_play_original['state'] = 'normal'
        btn_play_separado['state'] = 'normal'

    except Exception as e:
        messagebox.showerror("Processing Error", str(e))
def play_audio_thread():
    if audio_file:
        play_audio(audio_file)

def play_separado_thread():
    if output_file:
        play_audio(output_file)

def main():
    global ax_original, ax_separado, canvas

    root = tk.Tk()
    root.title("Bark Detection")
    root.geometry("800x600")

    btn_process = tk.Button(root, text="Select Audio File", command=process_audio)
    btn_process.pack(pady=20)

    global btn_play_original, btn_play_separado
    btn_play_original = tk.Button(root, text="Listen to Original Audio", state='disabled', command=lambda: threading.Thread(target=play_audio_thread).start())
    btn_play_original.pack(pady=5)

    btn_play_separado = tk.Button(root, text="Listen to Separated Audio", state='disabled', command=lambda: threading.Thread(target=play_separado_thread).start())
    btn_play_separado.pack(pady=5)

    fig, (ax_original, ax_separado) = plt.subplots(2, 1, figsize=(10, 8))

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    plt.tight_layout()  

    root.mainloop()

if __name__ == "__main__":
    main()
