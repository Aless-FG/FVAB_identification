from scipy import signal
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pywt
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import roc_curve, auc


def calculate_roc_auc_eer(subject, n_subjects, prediction): # funzione che calcola AUC e EER
    gt = np.zeros(n_subjects)
    for x in range(0, n_subjects):
        if x != subject:
            gt[x] = 0
        else:
            gt[x] = 1
    x = 0
    pred = np.zeros(n_subjects)
    for column in prediction.T:
        sumCol = 0
        avg = 0
        sumCol += sum(column)
        avg += sumCol / len(column)
        pred[x] = avg
        x = x + 1
    fpr, tpr, thresholds = roc_curve(gt, pred)
    auc_roc = auc(fpr, tpr)
    eer_roc = fpr[np.nanargmin(np.abs(tpr - (1 - fpr)))]
    return auc_roc, eer_roc


def preprocessing(n_subjects, n_sessions, n_stimuli, n_coeffwt, n_sample_psd):
    fs = 256  # frequenza di campionamento
    fc_hp = 0.5  # Frequenza di taglio inferiore
    fc_lp = 80  # Frequenza di taglio superiore
    N = 101
    session = 3
    for i in range(0, n_subjects):
        test_array = np.zeros((1, n_sessions, n_stimuli, n_coeffwt, n_sample_psd))
        s_mat = loadmat(f"/Users/macro/OneDrive/Desktop/Progetto Biometria/dataset/BED_Biometric_EEG_dataset/BED/RAW_PARSED/s{i + 1}_s{session}.mat")
        rec = s_mat['recording']
        events = s_mat['events']
        rec_dataframe = pd.DataFrame(rec,
                                     columns=['COUNTER', 'INTERPOLATED', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1',
                                              'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'UNIX_TIMESTAMP'])
        sensors = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        events_dataframe = pd.DataFrame(events, columns=['Start', 'End', 'Info'])
        start_array = np.concatenate(events_dataframe['Start'].values)
        end_array = np.concatenate(events_dataframe['End'].values)
        times = rec_dataframe['UNIX_TIMESTAMP'].values
        z = 0
        for k in range(0, len(start_array)):
            start_sample = np.argmin(np.where(times > start_array[k])[0])
            end_sample = np.argmax(np.where(times < end_array[k])[0])
            stimuli_eeg = []
            for sensor in sensors:
                sig = rec_dataframe[sensor]
                stimuli_eeg.extend(sig[start_sample:end_sample])
            h_lp = signal.firwin(N, fc_lp / (fs / 2), window='flattop')
            h_hp = signal.firwin(N, fc_hp / (fs / 2), window='flattop', pass_zero=False)
            # Applicazione del filtro passa basso al segnale x
            x_lp = signal.filtfilt(h_lp, 1, stimuli_eeg)
            # Applicazione del filtro passa alto al segnale x
            x_hp = signal.filtfilt(h_hp, 1, x_lp)
            wavelet = pywt.Wavelet('db4')
            coeffs = pywt.wavedec(x_hp, wavelet, level=4)
            y = 0
            for coeff in coeffs:
                f, psd = signal.welch(coeff, fs, window='blackman', nperseg=fs, noverlap=int(fs / 4))
                mirrored = np.concatenate((psd, psd[::-1]))
                test_array[0, 0, z, y, :] = mirrored
                y = y + 1
            z = z + 1
        n_sample = test_array.shape[1] * test_array.shape[2]
        new_shape = (test_array.shape[0] * n_sample, test_array.shape[3], test_array.shape[4])
        test_array = test_array.reshape(new_shape)
        labels_test = np.repeat(i, n_sample)
        y_test_cat = to_categorical(labels_test, num_classes=n_subjects)
        model = tf.keras.models.load_model(f'Model1/')
        y_scores = model.predict(test_array)
        roc_auc, eer = calculate_roc_auc_eer(i, n_subjects, y_scores)
        print(f"AUC for Subject {i + 1}:", roc_auc)
        print(f"EER for Subject {i + 1}:", eer)


if __name__ == "__main__":
    n_sub = 21
    n_ses = 1
    n_stim = 66
    n_coefwt = 5
    len_psd = 258
    preprocessing(n_sub, n_ses, n_stim, n_coefwt, len_psd)
