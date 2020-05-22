import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import pickle

DIR = "F:/ML/Deep learning using Tensorflow (Sentdex)/Cat vs Dog Classifier/TestData"
CATEGORIES = ["Dog", "Cat"]


IMG_SIZE = 70

testing_data = []
def create_test_data():
    
    for category in CATEGORIES:
        path = os.path.join(DIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)    
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([new_array, class_num])
            except Exception as e:
                pass

create_test_data()


random.shuffle(testing_data)

X_test = []
y_test = []

for feature, label in testing_data:
    X_test.append(feature)
    y_test.append(label)
    

X_test = np.array(X_test).reshape([-1, IMG_SIZE, IMG_SIZE, 1])
y_test = np.array(y_test)

pickle.dump(testing_data, open("testing_data.pickle", "wb"))
pickle.dump(X_test, open("X_test2.pickle", "wb"))
pickle.dump(y_test, open("y_test2.pickle", "wb"))









