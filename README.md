# Mnist
A feed forward neural network that gets a 93-94% accuracy on the mnist dataset
The idea behind this project has been to take many 28x28 pixels pictures of handwritten numbers and make a convolutinal neural network, that could determin which number there was written. The solution is based on taking the 28x28 pixels, and extract them to an array with 728 values. The model is then training on the values in each of these 728 "boxes" where it figures out a weight value, the weight value is how certain the model is on a specific number. At the start the weights are completely random, but with a cost function and optimization function patterns will be made, and the accuracy will rise. The code has been written on tensorflow version 1.8 with python 3.6, this is the GPU version of tensorflow for faster training times. 
To run this model, simply install tensorflow via pip: pip install tensorflow
This will get you the cpu version and the training time will not be too long.
