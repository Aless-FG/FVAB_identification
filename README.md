# FVAB_identification_verification
Subject identification and verification using EEG signals (Python3.8)

# Pre-requisiti
1) un PC decente altrimenti il preprocessing sarà molto lungo
2) Python 3.8 (non strettamente necessario, ma consigliato per garantire la massima compatibilità con i moduli)
3) dataset BED
4) IDE a vostra scelta
5) installazione librerie necessarie (vedasi sezione [Librerie](https://github.com/Aless-FG/FVAB_identification/tree/master#librerie))

# Identificazione
1) cambiare path dataset
2) runnare identification_test
3) fatto

# Verifica
1) cambiare path dataset
2) runnare verificationTraining
3) runnare verificationTest
4) fatto

# Verifica dei soggetti in base allo stimolo

I 66 stimoli a cui sono sottoposti sono così distribuiti nel dataset:

0-24 Stimoli Visivi

24-48 Stimoli Cognitivi

48-52 VEP

52-56 VEPC

56-64 Riposo

64 Occhi chiusi

65 Occhi aperti

Per ottenere quindi i risultati relativi all'AUC e all'EER è necessario:
1) Modificare nel main la variabile n_stim con il numero di stimoli (es. 24 nel caso di stimoli visivi o 8 nel caso di Riposo)
2) Nel terzo ciclo della funzione preprocessing inserire il range per il singolo stimolo come rappresentato sopra (es. nel caso di stimoli visivi range(0,24))
 
Queste operazioni vanno effettuate sia per VerificationTraining.py che per VerificationTest.py. 

# Librerie
`keras==2.12.`

`numpy==1.24.3`

`pandas==2.0.2`

`scikit_learn==1.2.2`

`scipy==1.10.1`

`tensorflow==2.12.0`

`PyWavelets==1.4.1`
