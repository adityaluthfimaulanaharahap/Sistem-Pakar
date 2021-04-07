# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:22:45 2021

@author: Aditya Luthfi
"""

import pandas
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

#baca csv
d = pandas.read_csv("iris_numeric_dataset.csv")

#acak data csv
d = d.sample(frac=1)

#membagi kolom atribute dan label dalam csv    
d_attribute = d.iloc[:, :4]
d_label = d.iloc[:, 4:]

#membagi data training dan data testing
d_train_attribute = d_attribute[:100]
d_train_label = d_label[:100]
d_test_attribute = d_attribute[100:]
d_test_label = d_label[100:]


gnb = GaussianNB()

# membuat model trainning
gnb.fit(d_train_attribute, d_train_label)

# memprediksi data test atribute dari model prediksi
Prediksi = gnb.predict(d_test_attribute) 

# akurasi prediksi yang dijalankan menggunakan gnb
Akurasi = gnb.score(d_test_attribute, d_test_label)

# memprediksi probabilitas
Prediksi_Probabilitas = gnb.predict_proba(d_test_attribute)

# visualisasi hasil prediksi menggunakan convusion matriks
pred_labels = gnb.predict(d_test_attribute)
cm = confusion_matrix(d_test_label, pred_labels)




