import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "F:/ML/Deep learning using Tensorflow (Sentdex)/Cat vs Dog Classifier/PetImages"

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

X_train = []
y_train = []

X_test = []
y_test = []

for feature, label in training_data[1:-51]:
    X_train.append(feature)
    y_train.append(label)
    
for feature, label in training_data[-50:len(training_data)]:
    X_test.append(feature)
    y_test.append(label)
    
    
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)

import pickle
pickle.dump(X_train, open("X_train.pickle", "wb"))
pickle.dump(y_train, open("y_train.pickle", "wb"))

pickle.dump(X_test, open("X_test.pickle", "wb"))
pickle.dump(y_test, open("y_test.pickle", "wb"))














