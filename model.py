# importing 
import pandas as pd
import numpy as np 

# loading the dataset
data_file = pd.read_csv("/home/kavi/Downloads/heart.csv")
x_data = data_file.iloc[:,:-1].values
y_data = data_file.iloc[:,-1:].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x_data)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y_data , test_size=0.2)



# model 
import tensorflow 
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout


machine = Sequential()

machine.add(Dense(units=13 , activation='relu', use_bias=True))
machine.add(Dense(units=26 , activation='relu',use_bias=True))
machine.add(Dropout(0.2))
machine.add(Dense(units=13 , activation='relu', use_bias=True))
machine.add(Dense(units=6 , activation='relu', use_bias=True))
machine.add(Dropout(0.2))
machine.add(Dense(units=3 , activation='relu', use_bias=True))
machine.add(Dense(units=1 , activation='sigmoid'))

machine.compile(optimizer="SGD", loss='binary_crossentropy',metrics='accuracy')
machine.fit(x_train, y_train , batch_size= 50, epochs= 1000)


machine.save("/home/kavi/Documents/GitHub/Heart_dis_deep_learning/heart_predict.h5")

