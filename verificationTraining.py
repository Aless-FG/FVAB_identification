
#import delle libreria necessarie
from scipy import signal
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pywt
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, LayerNormalization
import tensorflow as tf
from keras.utils.np_utils import to_categorical


def preprocessing(n_subject, n_sessions,n_stimuli, n_coeffWT, n_sample_psd): #funzione che effettua il preprocessing
    fs = 256  # frequenza di campionamento
    fc_hp = 0.5  # Frequenza di taglio inferiore
    fc_lp = 80  # Frequenza di taglio superiore
    N = 101 # paramtro che stabilisce il tipo di filtro (tipo I o tipo II)
    order = 4  # Ordine del filtro
    data_array = np.zeros((n_subject, n_sessions,n_stimuli, n_coeffWT, n_sample_psd)) # array multidimensionale
    for i in range(0,n_subject): # per ogni soggetto
        for j in range(0,n_sessions): # per le prime due sessioni
            s_mat = loadmat(f"/Users/macro/OneDrive/Desktop/Progetto Biometria/dataset/BED_Biometric_EEG_dataset/BED/RAW_PARSED/s{i+1}_s{j+1}.mat") # lettura file .mat
            rec = s_mat['recording'] # accesso colonna recording
            events = s_mat['events'] # accesso colonna events
            rec_dataframe = pd.DataFrame(rec, columns=['COUNTER', 'INTERPOLATED', 'F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4', 'UNIX_TIMESTAMP'])
            sensors = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'] #  canali eeg e timestamp
            events_dataframe = pd.DataFrame(events, columns=['Start', 'End', 'Info']) # inizio e fine recording
            start_array = np.concatenate(events_dataframe['Start'].values) # inizio
            end_array = np.concatenate(events_dataframe['End'].values) # fine
            times = rec_dataframe['UNIX_TIMESTAMP'].values # valori temporali
            z=0 # creazione variabile z settata a 0
            for k in range(0, n_stimuli): # per ogni stimolo
                start_sample = np.argmin(np.where(times > start_array[k])[0]) # inizio stimolo
                end_sample = np.argmax(np.where(times < end_array[k])[0]) # fine stimolo
                stimuli_eeg = [] # creazione array per gli stimoli
                for sensor in sensors: # per ogni sensore
                    sig = rec_dataframe[sensor] # accesso singolo canale
                    stimuli_eeg.extend(sig[start_sample:end_sample]) # concatenazione
                h_lp = signal.firwin(N, fc_lp / (fs / 2), window='flattop') # creazione filtro passa basso
                h_hp = signal.firwin(N, fc_hp / (fs / 2), window='flattop', pass_zero=False) # creazione filtro passa alto
                # Applicazione del filtro passa basso al segnale x
                x_lp = signal.filtfilt(h_lp, 1, stimuli_eeg)
                # Applicazione del filtro passa alto al segnale x
                x_hp = signal.filtfilt(h_hp, 1, x_lp)
                wavelet = pywt.Wavelet('db4') # trasformata wavelet discreta
                coeffs = pywt.wavedec(x_hp, wavelet, level=4) # applicazione trasformata
                y=0
                # applicazione PSD
                for coeff in coeffs:
                    f, psd = signal.welch(coeff, fs, window='blackman', nperseg=fs, noverlap=int(fs/4))
                    mirrored = np.concatenate((psd, psd[::-1])) # mirroring segnale
                    data_array[i, j, z, y, :] = mirrored
                    y=y+1
                z = z + 1
            print(f"Soggetto:{i+1}, Sessione:{j+1} completato. ")

    # reshaping array
    n_sample = data_array.shape[1]*data_array.shape[2]
    new_shape = (data_array.shape[0]*n_sample,data_array.shape[3],data_array.shape[4])
    data_reshape = data_array.reshape(new_shape)

    labels_train = np.repeat(np.arange(0,n_subject),n_sample) # etichette

    y_train_cat = to_categorical(labels_train,num_classes=n_subject) # onehot encoding
    # inizio modello
    model = Sequential()
    model.add(GRU(64, input_shape=(data_reshape.shape[1], data_reshape.shape[2]),
                  dropout=0.2,))
    model.add(LayerNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_subject, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary()) # stampa riassunto modello (com'è fatto)
    history = model.fit(data_reshape, y_train_cat, epochs=30, batch_size=32) # traning modello
    model.save("Model1/") # salvataggio modello


if __name__ == "__main__": # prima funzione ad essere eseguita
    n_sub = 21 # numero soggetti
    n_ses = 2 # numero sessioni
    n_stim = 66 # numero stimoli
    n_coefwt = 5 # numero coeff della wavelet
    len_psd = 258 # lunghezza psd
    preprocessing(n_sub, n_ses, n_stim, n_coefwt, len_psd) # chiamo la funzione che effettua il preprocessing e training