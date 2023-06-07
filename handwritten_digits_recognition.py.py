import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


if os.path.exists('model'):
    model = tf.keras.models.load_model('model')
else:
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(loss)
    print(accuracy)

    model.save('model')


for image in os.listdir('digits/'):
    path = os.path.join('digits/', image)

    if os.path.isfile(path):
        img = cv2.imread(path)[:,:,0]
        img = np.invert(np.array([img]))
        predict = model.predict(img)
        print("The Number is: ", np.argmax(predict))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()