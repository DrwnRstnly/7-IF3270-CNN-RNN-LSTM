<h1 align="center">Tugas Besar 2 IF3270 Pembelajaran Mesin</h1>
<h1 align="center">Kelompok 7</h3>
<h3 align="center">Convolutional Neural Network (CNN) dan Recurrent Neural Network (RNN)</p>

## Table of Contents

- [Abstraksi](#abstraksi)
- [Cara set up dan run](#cara-set-up-dan-run)
- [Pembagian Tugas](#pembagian-tugas)

## Abstraksi
Repository ini berisi implementasi dari Convolutional Neural Network (CNN) dan Recurrent Neural Network (RNN & LSTM) yang dibangun dari_scratch_. Pada bagian CNN, kode mencakup operasi konvolusi, pooling, fungsi aktivasi yang menggunakan relu dan softmax, serta proses forward dan backward propagation. Di sisi RNN & LSTM, terdapat implementasi sel recurrent dasar yang mampu memproses data sekuensial melalui unrolling time‐step, lengkap dengan mekanisme gradien balik. Sel RNN menggunakan aktivasi tanh pada update state dan sigmoid untuk gate sederhana. Sedangkan Sel LSTM mengimplementasikan tiga gate (forget, input, output) dengan sigmoid, serta cell-candidate menggunakan tanh. Output layer (jika klasifikasi) memakai softmax, serta proses forward dan backward propagation. Inti dari repository ini adalah memenuhi spesifikasi tugas besar 2 IF3270 Pembelajaran Mesin dengan menyoroti pemahaman mendalam tentang arsitektur CNN dan RNN.

## Cara set up dan run
1. Pastikan Python terinstal pada komputer yang akan menjalankan program di repository ini. Download atau instalasi Python dapat dilihat [disini](https://www.python.org/downloads/).
2. Kemudian clone repository ini dengan cara seperti berikut.
```
https://github.com/DrwnRstnly/7-IF3270-CNN-RNN-LSTM.git
cd 7-IF3270-CNN-RNN-LSTM
cd src
```
3. Lalu, gunakan code editor kesukaanmu yang dapat support notebook (.ipynb). Contohnya, `vscode`.
4. Jalankan semua file notebook `.ipynb` dengan fitur `Run All`. Dalam kasus ini adalah `cnn.ipynb`, `lstm.ipynb`, dan `rnn.ipynb`.
5. Seharusnya, program notebook sudah berjalan. **Ingat untuk menjalankan notebook mulai dari blok yang teratas, untuk mencegah error pada blok dibawahnya.**

**Note: Ingat untuk melakukan pip install terhadap library yang tidak tersedia pada environment Anda, dengan menggunakan command `pip install <nama_modul>`**<br>
**Modul atau library yang diperlukan dapat diinstal seperti pada `requirements.txt` dengan menjalankan `pip install -r /path/to/requirements.txt`**<br>
**Disarankan juga untuk menggunakan venv agar tidak konflik dengan library atau module yang sudah terinstal secara global**

## Membaca
Jika ingin melihat implementasi model dari scratch, maka implementasi dapat dilihat pada folder `src/classes` yang mana terdapat 3 jenis file yang mengandung implementasi forward propagation by scratch untuk masing-masing jenis model, yaitu `forward_cnn.py`, `forward_lstm.py`, dan `forward_rnn.py`.
## Pembagian Tugas
This project was developed by:
| NIM      | Nama                    | Kontribusi                                                                                                                                                                                                               |
|----------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 13522045 | Elbert Chailes           | Pembuatan model RNN                                                        |
| 13522047 | Farel Winalda    | Pembuatan model CNN                                                        |
| 13522115 | Derwin Rustanly    | Pembuatan model CNN & LSTM                                                  |
