from datetime import datetime
import numpy as np  
from sklearn.cluster import KMeans
import warnings

class TodoItem:
    def __init__(self, name, level, deadline):
        self.name = name
        self.level = level
        self.deadline = deadline

def convert_to_datetime(date_string):
    return datetime.strptime(date_string, '%d-%m-%Y')

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

data = get_todo_items()

num_clusters = 2

warnings.filterwarnings("ignore", category=FutureWarning)

kmeans = KMeans(n_clusters=num_clusters)

features = [[item.level + item.deadline.timestamp()] for item in data]

kmeans.fit(features)

centroids = kmeans.cluster_centers_

priorities = []
for feature in features:
    centroid_distances = [abs(feature - centroid) for centroid in centroids]
    if min(centroid_distances) == centroid_distances[0]:
        priorities.append("Prioritas")
    else:
        priorities.append("Nonprio")

for i, item in enumerate(data):
    print("Tugas:", item.name, "| Level:", item.level, "| Deadline:", item.deadline.strftime("%d-%m-%Y"), "| Prioritas:", priorities[i])
