import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "F:/ML/Deep learning using Tensorflow (Sentdex)/New folder/PetImages"

CATEGORIES = ["Dog", "Cat"]

training_data = []

IMG_SIZE = 70

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

import random
random.shuffle(training_data)

X = []
y = []

for feature, label in training_data:
    X.append(feature)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

import pickle
pickle.dump(X, open("X.pickle", "wb"))
pickle.dump(y, open("y.pickle", "wb"))
