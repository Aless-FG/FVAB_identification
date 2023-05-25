from scipy import signal
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pywt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tensorflow.python.estimator import keras


def resize_signals(signals, lenght):
    window_size = 20
    resized_signals = []
    for sig in signals:
        n_segments = int(len(sig) / window_size) + 1
        segments = []
        for seg in range(0, n_segments):
            if seg * window_size + window_size >= len(sig):
                start_idx = seg * window_size
                sig_seg = sig[start_idx:]
                pad = (seg + 1) * window_size - len(sig)
                sig_seg = np.pad(sig_seg, (0, pad), mode='constant', constant_values=0)
                segments.append(sig_seg)
            else:
                start_idx = seg * window_size
                end_idx = (seg + 1) * window_size
                sig_seg = sig[start_idx:end_idx]
                segments.append(sig_seg)
        energy = np.sum(np.array(segments) ** 2, axis=1)
        while True:
            min_energy_idx = np.argmin(energy)
            energy = np.delete(energy, min_energy_idx)
            start_idx = min_energy_idx * window_size
            if len(sig) == lenght:
                resized_signals.append(sig)
                break
            elif (len(sig) - window_size) < lenght:
                end_idx = start_idx + (len(sig) - lenght)
                sig = np.delete(sig, np.s_[start_idx:end_idx])
                resized_signals.append(sig)
                break
            else:
                end_idx = (min_energy_idx + 1) * window_size
                sig = np.delete(sig, np.s_[start_idx:end_idx])
    return np.array(resized_signals)


n_subjects = 1  # numero soggetti
n_sessions = 3  # numero sessioni
n_sensors = 14  # numero sensori
fs = 256  # frequenza di campionamento
fc_hp = 0.5  # Frequenza di taglio inferiore
fc_lp = 45  # Frequenza di taglio superiore
N = 101
reconstructed_model = tf.keras.saving.load_model("/home/ale/Desktop/model")
data_array = np.zeros((n_subjects, 1, 14, 129, 600))  # array multidimensionale

# da 1 a 3 (numero sessioni)
s_mat = loadmat(f"/home/ale/Desktop/BED_Biometric_EEG_dataset/BED/RAW_PARSED/s6_s3.mat")  # leggo i .mat
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
    psd = resize_signals(psd, 300)
    psd_norm = normalize(psd, 'l2')  # normalizzazione L2 del segnale
    mirrored = np.concatenate((psd_norm, psd_norm[::-1]), axis=1)
    data_array[0, 0, z, :, :] = mirrored
    z = z + 1

new_shape = (data_array.shape[0] * data_array.shape[1] * data_array.shape[2], data_array.shape[3], data_array.shape[4])
data_reshape = data_array.reshape(new_shape)
prediction = reconstructed_model.predict(data_reshape)

print("prediction shape:", prediction.shape)
print(prediction)
# predicted_labels = np.argmax(prediction, axis=1)
# print(predicted_labels)
subject = 1

for column in prediction.T:
    sumCol = 0
    avg = 0
    sumCol += sum(column)
    avg += sumCol / len(column)
    print("Soggetto:", subject, " ", str(avg))
    subject += 1
