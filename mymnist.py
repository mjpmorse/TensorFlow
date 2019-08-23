import numpy
import tensorflow as tf
from   tensorflow.keras import layers
#mnist dataset
mnist = tf.keras.datasets.mnist

#divide the data set into test and train sets
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



#build a model with three convolutional layers of 128 nodes
model = tf.keras.models.Sequential([
	layers.Flatten(input_shape(28,28)), # the mnist images are 28 by 28
	layers.Dense(128,activation='relu'), #first convolution layer will be a relu
	layers.Dense(128,activation='prelu'),
	layers.Dropout(0.2),
	layers.Dense(128,activation='relu'),
	layers.Dropout(0.1),
	layers.Dense(10,activation='softmax')])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)
model.evaluate(x_test,y_test)
