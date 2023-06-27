import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Membaca data dari file CSV
data = pd.read_csv("data_rumah.csv")

# Memisahkan fitur (ukuran rumah) dan target (harga rumah)
X = data["Ukuran"].values.reshape(-1, 1)
y = data["Harga"].values

# Membuat model regresi linier
model = LinearRegression()

# Melatih model dengan data
model.fit(X, y)

# Memasukkan ukuran rumah baru untuk diprediksi(masukkan angka beberapapun kedalam array maka akan generate prediksi harga)
ukuran_rumah_baru = np.array([10, 20, 30]).reshape(-1, 1)

# Melakukan prediksi harga rumah baru
harga_rumah_baru = model.predict(ukuran_rumah_baru)

# Menampilkan hasil prediksi
for ukuran, harga in zip(ukuran_rumah_baru, harga_rumah_baru):
    print(f"Ukuran Rumah: {ukuran} m^2, Prediksi Harga: {harga:.2f} juta")
