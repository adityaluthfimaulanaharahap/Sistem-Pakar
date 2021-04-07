Algoritma Naive Bayes memprediksi peluang di masa depan berdasarkan pengalaman di masa sebelumnya sehingga dikenal sebagai Teorema Bayes.

Disini saya menggunakan dataset iris dan melakukan prediksi menggunakan algoritma Naive Bayes. Prediksi disini bertujuan untuk memprediksi spesies dari bunga iris. Outputnya adalah jika :
- Nilai 0 merepresentasikan bunga iris spesies Setosa
- Nilai 1 merepresentasikan bunga iris Versicolor
- Nilai 2 merepresentasikan bunga iris Virginica

Penjelasan isi kode :

1. Pemanggilan library :
import pandas
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

2. Membaca file csv :
d = pandas.read_csv("iris_numeric_dataset.csv")

3. Mengacak urutan data dalam file csv untuk menyempurnakan data dalam melakukan klafisikasi :
d = d.sample(frac=1)

4. Memisahkan kolom attribute dan kolom label dari file csv. Kolom attribute ialah variable bebas dari dataset, dan kolom label ialah variable terikat dari dataset .Kolom attribute berisi 4 kolom pertama dari csv yaitu : sepal.length, sepal.width, petal.length, dan petal.width. Lalu kolom label berisi kolom ke-5 dari dataset yaitu variety
d_attribute = d.iloc[:, :4]
d_label = d.iloc[:, 4:]

5. Membagi data training dan data testing dari kolom attribute dan label
- Data training attribute dan label adalah 100 record pertama dari dataset 
- Data testing attribute dan label adalah record ke-101 dampai terakhir dari dataset

d_train_attribute = d_attribute[:100]
d_train_label = d_label[:100]
d_test_attribute = d_attribute[100:]
d_test_label = d_label[100:] 

6. Melakukan instansiasi kelas Naive Bayes
gnb = modelnb = GaussianNB()

7. Membuat model training untuk klasifikasi dari instansiasi kelas 
gnb.fit(d_train_attribute, d_train_label)

8. Memprediksi data test atribute dari model training yang dibuat.
Prediksi = gnb.predict(d_test_attribute)


9. Melihat akurasi dari prediksi yang dijalankan menggunakan Naive Bayes
Akurasi = gnb.score(d_test_attribute, d_test_label)

10. Memprediksi Probabilitas 
Prediksi_Probabilitas = gnb.predict_proba(d_test_attribute)

11. Melakukan visualisasi hasil prediksi menggunakan convusion matriks
pred_labels = gnb.predict(d_test_attribute)
cm = confusion_matrix(d_test_label, pred_labels)



