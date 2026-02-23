from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input((1,))
x = Dense(4, activation = 'relu')(inputs)
x = Dense(3, activation = 'selu')(x)
outputs = Dense(1, name = 'OutputLayer', activation = 'relu')(x)
model = Model(inputs, outputs, name = 'FCNN')
model.summary()
