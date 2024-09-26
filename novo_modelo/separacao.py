def separar_latidos(audio_file, output_file):
    # Carregar o áudio
    audio, sr = librosa.load(audio_file, sr=16000)

    # Calcular a energia do sinal
    energia = librosa.feature.rms(audio)[0]
    limiar = 0.02  # Ajuste conforme necessário
    latidos_presentes = energia > limiar

    # Obter índices dos latidos
    indices_latidos = np.where(latidos_presentes)[0]

    # Separar partes do áudio onde os latidos ocorrem
    latido_audio = []
    for idx in indices_latidos:
        start = max(0, idx - 1)  # 1 sample antes
        end = min(len(audio), idx + 1)  # 1 sample depois
        latido_audio.extend(audio[start:end])

    # Salvar o áudio separado
    sf.write(output_file, np.array(latido_audio), sr)

# Exemplo de uso
audio_file = '../dataset/dog/00e2b4cd.wav'
output_file = '../output/latido_separado.wav'
separar_latidos(audio_file, output_file)
print("Latidos separados e salvos em:", output_file)
