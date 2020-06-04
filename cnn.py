 #structuring the folder in terms of the outputs, to single out the 
#output classses for keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

#initialize CNN
classifier = Sequential()
#32 feature detectors of 3x3
#input size of image is 64*64
#ReLU function
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#classifier.add(Convolution2D(32,3,3,input_shape=(32,32,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#to optimize the accuracy on the test add another convolution layer
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))
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
classifier.add(Dropout(0.6))
classifier.add(Dense(output_dim = 128,activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(output_dim = 64,activation='relu'))
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
train_set = train_datagen.flow_from_directory(
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
fitted_model = classifier.fit(
           x = train_set,
           epochs=35,
           validation_data=test_set)

#print(fitted_model.history)
mean_accuracy = np.mean(fitted_model.history['acc'])
#print(mean_accuracy)

#make a single prediction cat=0 and dogs=1
#print(train_set.class_indices)

test_image = image.load_img('single_prediction/cat_or_dog_2.jpg', 
                            target_size=(64,64))
test_image = image.img_to_array(test_image)
#To make it compatible and make a batch of one input
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
#to find out if the output corresponds to cat or dog
if result[0][0] == 0:
    prediction = 'Cat'
else:
    predication = 'Dog'


