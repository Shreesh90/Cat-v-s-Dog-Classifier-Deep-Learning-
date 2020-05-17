import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = "2_conv(64,64)+0_Dense+Last_Dense_{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'logs\\{}'.format(NAME))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

#model = Sequential()
#
#model.add(Conv2D(256, (3,3), input_shape = X.shape[1:]) )
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))
#
#model.add(Conv2D(64, (3,3)))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))
#
#model.add(Conv2D(64, (3,3)))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))
#
#model.add(Flatten())
#
#model.add(Dense(1)) 
#model.add(Activation("sigmoid"))

model = tf.keras.models.load_model('Model_1(2_conv(64,64)+0_Dense+Last_Dense)')

model.compile(loss="binary_crossentropy",    #  binary_crossentropy
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X, y, batch_size=100, validation_split=0.10, epochs = 5, callbacks = [tensorboard])

model.save("Model_2(2_conv(64,64)+0_Dense+Last_Dense)") 

## classifier-6 (10 epochs) : main-conv + 2-conv(64,64) + 0-Dense + main-dense (Good Classifier)
