import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers.experimental import preprocessing
'''jeszcze nie wiesz co i jak'''
'''
data = pd.read_excel('dane.xlsx', sheet_name='mnozenie')
data = data.dropna()
x = data.loc[:,'x1':'x2']
y = data['y']

train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)
# x_train, x_test, y_train, y_test = train_test_split(x, y)

# sns.pairplot(train_dataset)
# plt.show()

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('y')
test_labels = test_features.pop('y')

n_features = train_features.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))
# compile the model
model.compile(optimizer='adam', loss='mse')
# fit the model
model.fit(train_features, train_labels, epochs=150, batch_size=32, verbose=0)
# evaluate the model
error = model.evaluate(test_features, test_labels, verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, np.sqrt(error)))



'''