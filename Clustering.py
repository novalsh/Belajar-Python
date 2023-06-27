from datetime import datetime
import numpy as np  
from sklearn.cluster import KMeans
import warnings

# Menggunakan kelas untuk menyimpan informasi tugas
class TodoItem:
    def __init__(self, name, level, deadline):
        self.name = name
        self.level = level
        self.deadline = deadline

# Mengonversi string tanggal ke objek datetime
def convert_to_datetime(date_string):
    return datetime.strptime(date_string, '%d-%m-%Y')

# Mengambil input dari pengguna
def get_todo_items():
    todo_items = []
    num_items = int(input("Masukkan jumlah tugas: "))
    for i in range(num_items):
        name = input("Nama tugas ke-{}: ".format(i+1))
        level = int(input("Level prioritas (1-5): "))
        deadline = input("Deadline (dd-mm-yyyy): ")
        deadline = convert_to_datetime(deadline)
        todo_items.append(TodoItem(name, level, deadline))
    return todo_items

# Data todolist dari input pengguna
data = get_todo_items()

# Jumlah cluster yang diinginkan
num_clusters = 2

# Menyembunyikan peringatan FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Membuat objek KMeans
kmeans = KMeans(n_clusters=num_clusters)

# Mengubah data todolist menjadi array fitur
features = [[item.level + item.deadline.timestamp()] for item in data]

# Melakukan proses clustering
kmeans.fit(features)

# Mendapatkan centroid dari setiap cluster
centroids = kmeans.cluster_centers_

# Menentukan prioritas berdasarkan perbandingan dengan centroid
priorities = []
for feature in features:
    centroid_distances = [abs(feature - centroid) for centroid in centroids]
    if min(centroid_distances) == centroid_distances[0]:
        priorities.append("Prioritas")
    else:
        priorities.append("Nonprio")

# Menampilkan hasil clustering dan prioritas
for i, item in enumerate(data):
    print("Tugas:", item.name, "| Level:", item.level, "| Deadline:", item.deadline.strftime("%d-%m-%Y"), "| Prioritas:", priorities[i])
