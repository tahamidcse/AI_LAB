


from tensorflow.keras.datasets.mnist import load_data

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
import keras



def build_model():
    number_of_classes=10
    inputs = Input((28,28,1),name='Input Layer')
    h1=Flatten()(inputs)
    
   
    h2 = Dense(32, activation='relu', name='h2')(h1)
    h3 = Dense(64, activation='relu', name='h3')(h2)
    h4 = Dense(128, activation='relu', name='h4')(h3)
    outputs = Dense(number_of_classes,activation='softmax', name = 'output_layer')(h4)
    model = Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model



def main():
    #--- Build model
    model=build_model()

    
    #.npz image to array save tf/lib/python3.10/site-packages/keras
    (trainX, trainY), (testX, testY) = load_data()
    trainY = to_categorical(trainY, num_classes = 10)
    testY = to_categorical(testY, num_classes = 10)
    
    

    
    model.fit(trainX, trainY, epochs=20, batch_size=32, validation_data=(testX, testY))
 

    model.summary(show_trainable = True)
    
    

    
if __name__ == "__main__":
  main()
