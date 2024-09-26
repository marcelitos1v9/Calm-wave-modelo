import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Função para carregar o dataset
def carregar_dados(dataset_path, sample_rate=22050, duration=5, fixed_length=216):
    data = []
    labels = []
    
    for label in os.listdir(dataset_path):
        label_dir = os.path.join(dataset_path, label)
        for audio_file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, audio_file)
            
            # Carregar o arquivo de áudio
            audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
            
            # Converter em um espectrograma de mel
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Garantir que todos os espectrogramas tenham o mesmo tamanho
            if mel_spectrogram_db.shape[1] < fixed_length:
                # Preencher com zeros se for menor que o tamanho fixo
                pad_width = fixed_length - mel_spectrogram_db.shape[1]
                mel_spectrogram_db = np.pad(mel_spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                # Truncar se for maior que o tamanho fixo
                mel_spectrogram_db = mel_spectrogram_db[:, :fixed_length]
            
            data.append(mel_spectrogram_db)
            labels.append(label)
    
    return np.array(data), np.array(labels)

# Caminho do dataset
dataset_path = './dataset/'

# Carregar dados
X, y = carregar_dados(dataset_path)

# Preprocessamento
X = X[..., np.newaxis]  # Adicionar dimensão extra para canal
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Definir o modelo CNN + RNN
def criar_modelo(input_shape, num_classes):
    model = models.Sequential([
        # Camadas convolucionais para extrair características
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Remodelar para entrada na RNN
        layers.Reshape((input_shape[0], -1)),
        
        # LSTM ou GRU para capturar dependências temporais
        layers.GRU(128, return_sequences=False),
        
        # Camadas densas para classificação
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Criar e treinar o modelo
input_shape = (X_train.shape[1], X_train.shape[2], 1)  # Especificar o shape do input
num_classes = len(np.unique(y_encoded))

modelo = criar_modelo(input_shape, num_classes)

# Callbacks (para visualizar o treinamento)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('modelo_cancelamento_ruido.h5', save_best_only=True)

# Treinar o modelo
historico = modelo.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, callbacks=[early_stopping, checkpoint])

# Visualizar gráfico de acurácia e perda
def plotar_resultados(historico):
    # Acurácia
    plt.plot(historico.history['accuracy'], label='Acurácia Treino')
    plt.plot(historico.history['val_accuracy'], label='Acurácia Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.show()

    # Perda
    plt.plot(historico.history['loss'], label='Perda Treino')
    plt.plot(historico.history['val_loss'], label='Perda Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()

# Plotar os resultados do treinamento
plotar_resultados(historico)
