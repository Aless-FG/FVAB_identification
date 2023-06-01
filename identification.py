from keras.metrics.confusion_metrics import Precision, Recall
from scipy import signal
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pywt
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import tensorflow as tf





n_subjects = 21  # numero soggetti
n_sessions = 3  # numero sessioni
n_sensors = 14  # numero sensori
fs = 256  # frequenza di campionamento
fc_hp = 0.5  # Frequenza di taglio inferiore
fc_lp = 45  # Frequenza di taglio superiore
N = 101

data_array = np.zeros((n_subjects, n_sessions, n_sensors, 129, 1606)) # array multidimensionale

for i in range(1, 22):  # da 1 a 21 (numero soggetti)
    for j in range(1, 4):  # da 1 a 3 (numero sessioni)
        s_mat = loadmat(f"/home/ale/Desktop/BED_Biometric_EEG_dataset/BED/RAW_PARSED/s{i}_s{j}.mat")  # leggo i .mat
        rec = s_mat['recording']  # accedo alla colonna recording
        rec_dataframe = pd.DataFrame(rec,
                                     columns=['COUNTER', 'INTERPOLATED', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1',
                                              'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'UNIX_TIMESTAMP'])
        sensors = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']  # sensori
        z = 0
        for sensor in sensors:  # ciclo i sensori
            sig = rec_dataframe[sensor]  # prendo il segnale del sensore
            h_lp = signal.firwin(N, fc_lp / (fs / 2), window='flattop')  # creazione filtro passa basso
            h_hp = signal.firwin(N, fc_hp / (fs / 2), window='flattop', pass_zero=False)  # creazione filtro passa alto

            x_lp = signal.filtfilt(h_lp, 1, sig)  # Applicazione del filtro passa basso al segnale x

            x_hp = signal.filtfilt(h_hp, 1, x_lp)  # Applicazione del filtro passa alto al segnale x

            wavelet = pywt.Wavelet('db4')  # trasformata Wavelet
            coeffs = pywt.wavedec(x_hp, wavelet, level=2)  # applico trasformata Wavelet a 2 livelli al segnale filtrato

            # Esempio di calcolo della TD-PSD su una sub-banda di frequenza utilizzando la finestra di blackman
            f, t_psd, psd = signal.spectrogram(coeffs[0], fs, window='blackman', nperseg=fs, noverlap=int(fs / 4))
            psd = tf.keras.utils.pad_sequences(psd, maxlen=803, dtype="float32", padding="pre") # padding del segnale
            psd_norm = normalize(psd, 'l2')  # normalizzazione L2 del segnale
            mirrored = np.concatenate((psd_norm, psd_norm[::-1]), axis=1)

            data_array[i - 1, j - 1, z, :, :] = mirrored
            z = z + 1

new_shape = (data_array.shape[0] * data_array.shape[1] * data_array.shape[2], data_array.shape[3], data_array.shape[4])
data_reshape = data_array.reshape(new_shape)

# crea un array numpy di valori interi da 1 a 21
labels_train = np.repeat(np.arange(0, 21), 42) # 3 * 14 = 42 (sessioni per sensori)
# Suddivisione dati in training set e test set
X_train, X_test, y_train, y_test = train_test_split(data_reshape, labels_train, test_size=0.3, random_state=42,
                                                    shuffle=True)
y_train_cat = to_categorical(y_train, num_classes=21)
y_test_cat = to_categorical(y_test, num_classes=21)

# modello LSTM
model = Sequential()
model.add(GRU(64, input_shape=(129, 1606), dropout=0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(21, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', Precision(), Recall()])
print(model.summary())
history = model.fit(X_train, y_train_cat, epochs=15, validation_split=0.2, batch_size=32)
print(model.evaluate(X_test, y_test_cat))
model.get_metrics_result()

#model.save("/home/ale/Desktop/model")