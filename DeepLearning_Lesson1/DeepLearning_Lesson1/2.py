import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.model_selection import train_test_split
dataset = pd.read_csv("Breas Cancer.csv")
dataset.diagnosis=dataset.diagnosis.replace({'M':1,'B':0})


import numpy as np
dataset=dataset.values
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:-1], dataset[:,1], test_size=0.25, random_state=87)
np.random.seed(187)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(78, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0, initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test,verbose=2))
