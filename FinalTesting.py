from tensorflow import keras
import pickle
import matplotlib.pyplot as plt

Training_data = pickle.load(open("Training_data.pickle", "rb"))
X_test = pickle.load(open("X_test.pickle", "rb"))
y_test = pickle.load(open("y_test.pickle", "rb"))

X_test = X_test/255.0

classifier = keras.models.load_model('Model_3(2_conv(64,64)+0_Dense+Last_Dense)')

test_loss, test_accuracy = classifier.evaluate(X_test, y_test, verbose=2)

predictions = classifier.predict(X_test)


counter = 0
for i in range(len(X_test)):
    if predictions[i][0] > 0.5 and y_test[i] == 0 :
        plt.imshow(Training_data[len(Training_data)-50+i][0], cmap="gray")
        plt.show()
        print(predictions[i][0])
        counter += 1

    elif predictions[i][0] <= 0.5 and y_test[i] == 1 :
        plt.imshow(Training_data[len(Training_data)-50+i][0], cmap="gray")
        plt.show()
        print(predictions[i][0])
        counter += 1

print(counter)
    
    
    