from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model


inputs = Input((224, 224, 3))
x = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(inputs)
x = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D()(x) # Downsampling 112
x = Conv2D(filters = 128, kernel_size = (3, 3),padding = 'same', activation = 'relu')()
x = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D()(x) # Downsampling 56
x = Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
x = Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
x = Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D()(x) # Downsampling 28
x = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
x = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
x = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D()(x) # Downsampling 14
x = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
x = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
x = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same', activation = 'relu')(x)
x = MaxPooling2D()(x) #DownSampling7
x = Flatten()(x)
x = Dense(4096, activation = 'relu')(x)
x = Dense(4096, activation = 'relu')(x)
outputs = Dense(1000, name = 'OutputLayer', activation = 'softmax')(x)
model = Model(inputs, outputs, name = 'CNN')
model.summary()
