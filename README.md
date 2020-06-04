# PetIdentify
#### Build a Convolutional Neural Net to predict animal associated with a given picture.

#### The images consists of 2 classes - Dogs and Cats. The image folders for the training and test sets are structured according to the two classes. Out of a total of 10000 images, 8000 in the training set and 2000 for the test set. 

#### A key aspect to building and fitting the model was to implement image augmentation to the training set of images in order to
      1. Enrich the Dataset.
      2. Add different variations to the images.
      
#### The layers consists of 4 Convolution layers along with 3 Dense layers.

#### In order to counter overfitting two dropout layers were added in.

#### So far, an accuracy of 86% is being achieved on the test set with a batch size of 32 and 35 epochs.


      
