
from keras import regularizers
from keras.metrics.confusion_metrics import Precision, Recall
from scipy import signal
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pywt
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, LayerNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import tensorflow as tf





n_sessions=3
max=0

fs = 256  # frequenza di campionamento
fc_hp = 0.5  # Frequenza di taglio inferiore
fc_lp = 80  # Frequenza di taglio superiore
N = 101 # paramtro che stabilisce il tipo di filtro (tipo I o tipo II)
order = 4  # Ordine del filtro
data_array = np.zeros((21, n_sessions,66, 5, 258)) # array multidimensionale
for i in range(1,22):
    for j in range(1,4):
        s_mat = loadmat(f"/home/ale/Desktop/BED_Biometric_EEG_dataset/BED/RAW_PARSED/s{i}_s{j}.mat") # lettura file dataset
        rec = s_mat['recording']
        events = s_mat['events']
        rec_dataframe = pd.DataFrame(rec, columns=['COUNTER', 'INTERPOLATED', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'UNIX_TIMESTAMP'])
        sensors = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4']
        events_dataframe = pd.DataFrame(events, columns=['Start', 'End', 'Info'])
        start_array = np.concatenate(events_dataframe['Start'].values)
        end_array = np.concatenate(events_dataframe['End'].values)
        times = rec_dataframe['UNIX_TIMESTAMP'].values
        z=0
        for k in range(0, 66):
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
            # applicazione trasformata wavelet discreta
            wavelet = pywt.Wavelet('db4')
            coeffs = pywt.wavedec(x_hp, wavelet, level=4)

            y=0
            for coeff in coeffs:
                f, psd = signal.welch(coeff, fs, window='blackman', nperseg=fs, noverlap=int(fs/4)) # calcolo della PSD

                mirrored = np.concatenate((psd, psd[::-1])) # mirroring del segnale
                data_array[i - 1, j-1, z, y, :] = mirrored
                y=y+1
            z = z + 1


        print(f"Soggetto:{i}, Sessione:{j} complete. ") # dov'è arrivato il preprocessing?


n_labels = data_array.shape[1]*data_array.shape[2]
new_shape = (data_array.shape[0]*n_labels,data_array.shape[3],data_array.shape[4])
data_reshape = data_array.reshape(new_shape)
# crea un array numpy di valori interi rappresentante i soggetti e il numero che identifica questi ultimi
labels_train = np.repeat(np.arange(0,21),n_labels)
X_train, X_test, y_train, y_test = train_test_split(data_reshape, labels_train, test_size=0.3, random_state=42,
                                                    shuffle=True)
# Suddivisione dati in training set e test set
y_train_cat = to_categorical(y_train,num_classes=21) #onehot encoding
y_test_cat = to_categorical(y_test,num_classes=21) #onehot encoding


for i in range(0,11): # 11 test

    model = Sequential()
    model.add(GRU(64, input_shape=(data_reshape.shape[1], data_reshape.shape[2]),
                  dropout=0.2,
                  ))
    """deCommentare le successive 2 righe per usare una LSTM"""
    # model.add(LSTM(32,recurrent_dropout=0.1,dropout=0.2))return_sequences=True,
    # model.add(Dense(32, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dense(16, activation='relu')) #Aggiunto layer di attivazione RELU
    model.add(Dense(21, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall()])
    # early stop con controllo sulla validation loss con un intervallo di controllo lungo 15 epoche
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min')
    print(model.summary())
    history = model.fit(X_train, y_train_cat, epochs=150, batch_size=32, callbacks=early_stop,validation_split=0.2)
    # Test del modello con stampa di validation loss, accuracy, precision e recall
    print
    print(model.evaluate(X_test, y_test_cat))


