import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
from keras.utils import normalize


if os.path.exists('model.keras'):
    model = load_model('model.keras')
else:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = normalize(X_train, axis=1)
    X_test = normalize(X_test, axis=1)

    model = Sequential([
        Flatten(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    model.save('model.keras')


for image in os.listdir('digits/'):
    path = os.path.join('digits/', image)

    if os.path.isfile(path):
        img = cv2.imread(path)[:,:,0]
        img = np.invert(np.array([img]))
        predict = model.predict(img)
        print("The Number is: ", np.argmax(predict))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
