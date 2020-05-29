 #structuring the folder in terms of the outputs, to single out the 
#output classses for keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#initialize CNN
classifier = Sequential()
#32 feature detectors of 3x3
#input size of image is 64*64
#ReLU function
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())

#classifier.summary()
#understand the layers better
# =============================================================================
# for layers in classifier.layers:
#     if 'conv' not in layers.name:
#         continue
#     filter,bias = layers.get_weights()
#     print(layers.name,filter.shape)
# =============================================================================
#Good practice- take powers of 2 for hidden layers
classifier.add(Dense(output_dim = 128,activation='relu'))
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

#Fit the images-image augmentation to prevent overfitting- Check 
#General Notes file for more info
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#class_mode is binary as there are two possible outcomes
train_set = test_datagen.flow_from_directory(
           'training_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

test_set = test_datagen.flow_from_directory(
           'test_set',
           target_size=(64, 64),
           batch_size=32,
           class_mode='binary')

#fit on train set and test performance on the test set
classifier.fit(
           train_set,
           steps_per_epoch=8000,
           epochs=25,
           validation_data=test_set,
           validation_steps=2000)


