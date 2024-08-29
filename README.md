# Laporan Machine Learning Terapan Dicoding
## Submission 1: Diabetes Classification

![diabetes-and-healthy-lifestyle](https://github.com/user-attachments/assets/f921062a-a667-4f44-8bb8-3384704bd976)

## Nama : Muhammad Islahfari Wahid

# Domain Proyek
Diabetes mellitus adalah penyakit metabolisme yang kronis yang mana pesien penyakit diabetes tidak menghasilkan jumlah insulin yang cukup atau bisa dikatakan tubuh tidak sanggup memanfaatkan insulin dengan baik sehingga menyebabkan gula darah di dalam tubuh mengalami jumlah yang berlebihan, kondisi ini sering kali dirasakan setelah komplikasi terjadi pada organ tubuh [1]. Merajuk pada data Federasi Diabetes Internasional, diprediksi penderita penyakit diabetes di Indonesia akan bertambah menjadi 16.2 juta pada tahun 2040 [2]. Dengan jumlah penderita yang meningkat tiap tahunnya sehingga dibutuhkan sistem deteksi dini untuk dapat melakukan pencegahan awal. Dibutuhkan pembuatan Model Machine Learning untuk membantu melakukan prediksi terhadap penderita penyakit diabetes.

# Business Understanding
## Problem Statements
Berdasarkan domain proyek di atas, berikut merupakan problem statement yang akan dilakukan:
* Bagaimana membuat model machine learning menggunakan data medis untuk mengetahui seseorang memiliki resiko peyakit diabetes?
* Bagaimana dapat menentukan model yang optimal untuk kebutuhan klasifikasi?

## Goals
* Membuat model untuk dapat memprediksi data medis pasien yang beresiko terkena penyakit diabetes
* Melakukan perbandingan evaluasi model untuk menentukan model terbaik

## Solution Statements
Membangun model dengan menggunakan dua jenis algoritma yang berbeda untuk dapat melakukan perbandingan kinerja model yang lebih baik

# Data Understanding
## Dataset
Dataset yang digunakan merupakan dataset yang diambil dari Kaggle, Dataset dapat diakses pada link berikut https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset.
![image](https://github.com/user-attachments/assets/9644421f-147b-4f5a-ad60-dc89048a724e)

Pada Dataset terdapat 100000 data dan terdiri dari 9 kolom dengan keterangan sebagai berikut. sebagai berikut gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes.
![image](https://github.com/user-attachments/assets/7cfa1766-fc3f-4058-9048-f0de5a0dde10)

1. **gender** : Jenis kelamin
2. **age** : Umur
3. **hypertension** : Kondisi tekanan darah terus meningkan pada arteri
4. **heart_disease** : Penyakit jantung
5. **smoking_history** : Riwayat perokok
6. **bmi** : Ukuran lemak tubuh berdasarkan berat dan tinggi badan
7. **HbA1c_level** : Ukuran kadar gula darah rata-rata seseorang selama 2-3 bulan terakhir
8. **blood_glucose_level** : Jumlah glukosa dalam aliran darah pada waktu tertentu
9. **diabetes** : Resiko penderita diabetes

## Types Data
![image](https://github.com/user-attachments/assets/524b72be-0e50-42cb-a6e7-eed038a365e3)

Selanjutnya dilakukan pengecekan tipe data dari dataset, untuk mengetahui kolom categorical dan numerical.

## Exploratory Data Analysis (EDA)
Melakukan visualisasi data categorical untuk mengetahui perbandingan persentase data tiap kategori.
![image](https://github.com/user-attachments/assets/c186f56e-a162-44ae-8932-7eadbb6c6216)

Pada gambar dapat dilihat untuk kolom gender memiliki persentase yang cukup mirip, pada kolom heart_disease dan hypertension terlihat cukup jauh perbandingan presentasi data yang dimiliki. Terakhir untuk smoking_history data terbanyak untuk kategori No Info & Never.

![Untitled](https://github.com/user-attachments/assets/0bea36d9-46df-4363-b42f-ca5c66b80077)

Selanjutnya melakukan visualisasi distribusi data untuk data numerical. Pada gambar dapat dilihat untuk data numerical cukup baik pada kolom age dan bmi.

![Untitled](https://github.com/user-attachments/assets/a4faf8af-9225-4143-8ce6-e2ed131c12cc)

Kemudian dilakukan visualisasi korelasi pada tiap kolom, untuk melihat hubungan antar kolom feature dan kolom target. Dapat dilihat kolom yang memiliki korelasi cukup tinggi pada diabetes adalah HbA1c_level dan blood_glucose_level

# Data Preparation
## Null Values
![image](https://github.com/user-attachments/assets/191f5ad6-aa0e-4dad-b3e4-f7ffbc1d50c7)

Karena dataset tidak memiliki nilai null sehingga tidak perlu dilakukan penanganan missing value, namun jika memiliki nilai null dapat melakukan berbagai metode penanganan missing value salah satunya dengan menggunakan mean.

## Categorical Encoding
![image](https://github.com/user-attachments/assets/d9e8ce3b-5e81-4442-940e-a08b2a3c4805)

Pada dataset data categorical diubah menjadi value bernilai angka untuk memudahkan pemrosesan model Machine Learning, salah satu teknik yang digunakan adalah metode Categorical Encoding

## Imbalance Label
![image](https://github.com/user-attachments/assets/eec4eea5-911a-4423-9d83-44bc79ce5828)

Dataset memiliki imbalance label yang terlalu besar, dikhawatirkan model nantinya akan lebih cenderung memprediksi label yang lebih banyak. Sehingga perlu dilakukan balanced label dengan cara mengambil sample secara acak dari data yang sudah dimiliki agar jumlah data label menjadi sama.

## Split Data
![image](https://github.com/user-attachments/assets/16feafdc-f4f5-42a7-8e73-c60dd7951f67)

Pembagian dataset dilakukan menjadi dua yaitu data training dan data test dengan skala perbandingan 80:20. Data training digunakan untuk pelatihan model sedangkan data test digunakan untuk evaluasi kinerja model.

## Normalization Dataset
![image](https://github.com/user-attachments/assets/be3bb4a4-d775-4119-8054-43dbfcacd355)

Normalisasi data dilakukan dengan tujuan untuk menyesuaikan skala data sehingga tidak terjadi bias pada feature yang akan di proses oleh model. Tahap ini mengubah data menjadi skala 0 sampai 1
