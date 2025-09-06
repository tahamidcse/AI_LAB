


from tensorflow.keras.datasets.mnist import load_data

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
import keras
import os



def build_model():
    number_of_classes=10
    inputs = Input((28,28,1),name='Input Layer')
    h1=Flatten()(inputs)
    
   
    h2 = Dense(32, activation='relu', name='h2')(h1)
    h3 = Dense(64, activation='relu', name='h3')(h2)
    h4 = Dense(128, activation='relu', name='h4')(h3)
    outputs = Dense(number_of_classes,activation='softmax', name = 'output_layer')(h4)
    model = Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model



def main():
    #--- Build model
    model=build_model()

    
    #.npz image to array save tf/lib/python3.10/site-packages/keras
    (trainX, trainY), (testX, testY) = load_data()
    cwd = os.getcwd()
    npz_path = os.path.join(cwd, "tf", "lib", "python3.10", "site-packages", "keras", "datasets", "my_dataset.npz")

    
    data    = np.load(npz_path)
    X_train = data['x_train']
    X_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    trainX = trainX.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    testX = testX.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    X_train = X_train.astype("float32")/255.0
    X_test  = X_test.astype("float32")/255.0
    trnX = np.concatenate((trainX, X_train), axis=0)
    trnY = np.concatenate((trainY, y_train), axis=0)
    tstX = np.concatenate((testX, X_test), axis=0)
    tstY = np.concatenate((testY, y_test), axis=0)
    
    

    
    model.fit(trnX, trnY, epochs=20, batch_size=32, validation_data=(tstX, tstY))
 

    model.summary(show_trainable = True)
    
    

    
if __name__ == "__main__":
  main()
