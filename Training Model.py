import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D, Dropout
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = "2_conv(64,64)+0_Dense+Last_Dense_{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'logs\\{}'.format(NAME))

X_train = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))

X_train = X_train/255.0

model = Sequential()

model.add(Conv2D(256, (3,3), input_shape = X_train.shape[1:]) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2, input_shape = X_train.shape[1:]))
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2, input_shape = X_train.shape[1:]))
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1)) 
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",    #  binary_crossentropy
              optimizer="adam",
              metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=100, validation_split=0.10, epochs = 5, callbacks = [tensorboard])

model.save("Model_5(2_conv(64,64)+0_Dense+Last_Dense)") 


