import numpy as np
import pandas as pd
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
import keras.backend as K
from keras.models import Sequential
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D,AveragePooling2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def keras_model(image_x, image_y):
    num_of_classes = 6
    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=(image_X, image_Y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(128, (5, 5), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "emojinator_model.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list

def main():
    data = pd.read_csv("images.csv")
    dataset = np.array(data)
    np.random.shuffle(dataset)
    X = dataset
    Y = dataset
    X = X[:, 1:2501]
    Y = Y[:, 0]

    X_train = X[0:12000, :]
    X_train = X_train / 255.
    X_test = X[12000:13201, :]
    X_test = X_test / 255.

    # Reshape
    Y = Y.reshape(Y.shape[0], 1)
    Y_train = Y[0:7000, :]
    Y_train = Y_train.T
    Y_test = Y[7000:7800, :]
    Y_test = Y_test.T

    
    image_x = 50
    image_y = 50

    train_y = np_utils.to_categorical(Y_train)
    test_y = np_utils.to_categorical(Y_test)
    train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
    test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
    X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
    X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
    
    model, callbacks_list = keras_model(image_x, image_y)
    model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=10, batch_size=64,
              callbacks=callbacks_list)
    scores = model.evaluate(X_test, test_y, verbose=0)
    print("CNN Error: %.2f%%" % (100*(1 - scores[1] )))

    model.save('emojinator_model.h5')


main()
