import os
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from scipy.io.wavfile import write
from librosa.display import specshow
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Função para capturar áudio em tempo real
def capturar_audio(sample_rate=22050, duration=5):
    print("Capturando áudio por", duration, "segundos...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Esperar o áudio ser capturado
    return np.squeeze(audio)

# Função para gerar o espectrograma de mel
def gerar_melspectrogram(audio, sample_rate=22050):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

# Função para carregar o modelo treinado
def carregar_modelo(modelo_path='modelo_cancelamento_ruido.h5'):
    return tf.keras.models.load_model(modelo_path)

# Função para redimensionar o espectrograma
def redimensionar_espectrograma(mel_spectrogram, target_shape=(128, 216)):
    if mel_spectrogram.shape[1] < target_shape[1]:
        # Padroniza o espectrograma adicionando zeros
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, target_shape[1] - mel_spectrogram.shape[1])), mode='constant')
    elif mel_spectrogram.shape[1] > target_shape[1]:
        # Reduz o espectrograma cortando
        mel_spectrogram = mel_spectrogram[:, :target_shape[1]]
    return mel_spectrogram

# Função para identificar o som capturado
def identificar_som(modelo, mel_spectrogram):
    mel_spectrogram = redimensionar_espectrograma(mel_spectrogram)  # Redimensionar espectrograma
    mel_spectrogram = mel_spectrogram[np.newaxis, ..., np.newaxis]  # Ajustar para formato esperado pelo modelo
    predicao = modelo.predict(mel_spectrogram)
    return np.argmax(predicao, axis=1)[0]  # Retorna a classe prevista

# Função para aplicar um filtro de redução de ruído na faixa de frequência específica
def reduzir_ruido(audio, sample_rate=22050, frequencia_foco=(1000, 3000)):
    # Aplicar um filtro de atenuação para a faixa de frequências identificada
    fft = np.fft.rfft(audio)
    frequencias = np.fft.rfftfreq(len(audio), 1/sample_rate)
    
    # Atenuar frequências dentro da faixa identificada
    mask = (frequencias >= frequencia_foco[0]) & (frequencias <= frequencia_foco[1])
    fft[mask] = 0  # Zerar os valores da FFT para remover o ruído
    audio_filtrado = np.fft.irfft(fft)
    return audio_filtrado

# Função para plotar o espectrograma
def plotar_espectrograma(ax, mel_spectrogram_db, title):
    ax.clear()
    specshow(mel_spectrogram_db, sr=22050, ax=ax, cmap='viridis')
    ax.set(title=title)
    ax.set(xlabel='Tempo', ylabel='Frequência')
    plt.draw()

# Função principal que conecta a interface e o processamento
def executar_com_interface():
    modelo = carregar_modelo('modelo_cancelamento_ruido.h5')
    
    # Função interna para capturar e processar o áudio
    def processar_audio():
        # Capturar o áudio
        audio = capturar_audio()
        
        # Gerar espectrograma antes da filtragem
        mel_spectrogram_original = gerar_melspectrogram(audio)
        plotar_espectrograma(ax1, mel_spectrogram_original, 'Espectrograma Original')
        
        # Identificar o som
        classe_identificada = identificar_som(modelo, mel_spectrogram_original)
        
        # Exibir a classe identificada
        messagebox.showinfo("Classe Identificada", f"A classe identificada é: {classe_identificada}")
        print(f"Classe identificada: {classe_identificada}")  # Exibir no console

        # Aplicar a redução de ruído conforme a classe identificada
        if classe_identificada == 0:  # Ambulância
            audio_filtrado = reduzir_ruido(audio, frequencia_foco=(1000, 3000))
        elif classe_identificada == 1:  # Cachorro
            audio_filtrado = reduzir_ruido(audio, frequencia_foco=(200, 1000))
        elif classe_identificada == 2:  # Bombeiros
            audio_filtrado = reduzir_ruido(audio, frequencia_foco=(500, 4000))
        elif classe_identificada == 3:  # Tráfego
            audio_filtrado = reduzir_ruido(audio, frequencia_foco=(100, 800))
        else:
            audio_filtrado = audio  # Nenhum som identificado

        # Gerar espectrograma depois da filtragem
        mel_spectrogram_filtrado = gerar_melspectrogram(audio_filtrado)
        plotar_espectrograma(ax2, mel_spectrogram_filtrado, 'Espectrograma Filtrado')
        
        # Reproduzir o áudio filtrado
        sd.play(audio_filtrado, samplerate=22050)
        sd.wait()

    # Função para escolher um arquivo de áudio (opcional)
    def escolher_arquivo():
        arquivo_audio = filedialog.askopenfilename(title="Selecionar arquivo de áudio", filetypes=[("WAV files", "*.wav")])
        if arquivo_audio:
            messagebox.showinfo("Arquivo Selecionado", f"Arquivo {os.path.basename(arquivo_audio)} foi selecionado.")
            
            # Processar o áudio do arquivo selecionado
            audio, _ = librosa.load(arquivo_audio, sr=22050)
            mel_spectrogram_original = gerar_melspectrogram(audio)
            plotar_espectrograma(ax1, mel_spectrogram_original, 'Espectrograma Original')
            
            # Identificar o som
            classe_identificada = identificar_som(modelo, mel_spectrogram_original)
            
            # Exibir a classe identificada
            messagebox.showinfo("Classe Identificada", f"A classe identificada é: {classe_identificada}")
            print(f"Classe identificada: {classe_identificada}")  # Exibir no console
            
            # Aplicar a redução de ruído conforme a classe identificada
            if classe_identificada == 0:  # Ambulância
                audio_filtrado = reduzir_ruido(audio, frequencia_foco=(1000, 3000))
            elif classe_identificada == 1:  # Cachorro
                audio_filtrado = reduzir_ruido(audio, frequencia_foco=(200, 1000))
            elif classe_identificada == 2:  # Bombeiros
                audio_filtrado = reduzir_ruido(audio, frequencia_foco=(500, 4000))
            elif classe_identificada == 3:  # Tráfego
                audio_filtrado = reduzir_ruido(audio, frequencia_foco=(100, 800))
            else:
                audio_filtrado = audio  # Nenhum som identificado

            # Gerar espectrograma depois da filtragem
            mel_spectrogram_filtrado = gerar_melspectrogram(audio_filtrado)
            plotar_espectrograma(ax2, mel_spectrogram_filtrado, 'Espectrograma Filtrado')
            
            # Reproduzir o áudio filtrado
            sd.play(audio_filtrado, samplerate=22050)
            sd.wait()

    # Interface gráfica com Tkinter
    root = tk.Tk()
    root.title("Cancelamento de Ruído em Tempo Real")
    
    # Configuração dos gráficos para espectrogramas
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Botões de controle
    btn_capturar = tk.Button(root, text="Capturar e Processar Áudio", command=processar_audio)
    btn_capturar.pack(side=tk.LEFT, padx=10, pady=10)
    
    btn_escolher_arquivo = tk.Button(root, text="Escolher Arquivo de Áudio", command=escolher_arquivo)
    btn_escolher_arquivo.pack(side=tk.LEFT, padx=10, pady=10)

    tk.mainloop()

# Executar a interface gráfica
executar_com_interface()
